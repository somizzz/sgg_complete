import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysgg.modeling.utils import cat
from pysgg.modeling import registry
from pysgg.modeling.poolers import Pooler
from pysgg.modeling.make_layers import group_norm
from pysgg.modeling.make_layers import make_fc
from pysgg.structures.boxlist_ops import boxlist_union
from pysgg.modeling.roi_heads.box_head.inference import (
    PostProcessor,BoxCoder
)
from pysgg.modeling.roi_heads.box_head.loss import FastRCNNLossComputation
from pysgg.modeling.roi_heads.box_head.box_head import (
    add_predict_info,add_predict_logits
)
from pysgg.modeling.matcher import Matcher
from pysgg.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from pysgg.modeling.roi_heads.box_head.sampling import FastRCNNSampling
from pysgg.modeling.roi_heads.box_head.roi_box_feature_extractors import (
    ResNet50Conv5ROIFeatureExtractor
)

from pysgg.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    gt_rel_proposal_matching,
    # RelationProposalModel,
    filter_rel_pairs,
    MultiHeadAttention,
)
from pysgg.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)
from pysgg.modeling.roi_heads.relation_head.roi_relation_predictors import (
    obj_prediction_nms
)
from pysgg.utils.global_buffer import store_data
from pysgg.structures.boxlist_ops import boxlist_iou, squeeze_tensor
from pysgg.structures.bounding_box import BoxList
from pysgg.modeling.roi_heads.relation_head.loss import RelationLossComputation
from pysgg.modeling.roi_heads.relation_head.utils_motifs import (
    encode_box_info,
    obj_edge_vectors,
)
#from util.load_word2vec import obj_edge_vectors
from util.load_bert import obj_edge_vectors_bert

from pysgg.modeling.roi_heads.relation_head.utils_relation import (
    get_box_info,
    get_box_pair_info,
    layer_init,
)
from pysgg.modeling.roi_heads.relation_head.rel_proposal_network.loss import (
    FocalLoss,
    FocalLossFGBGNormalization,
    WrappedBCELoss,
    loss_eval_mulcls_single_level,
    loss_eval_hybrid_level,
    loss_eval_bincls_single_level
    # RelAwareLoss
)
from pysgg.modeling.roi_heads.relation_head.model_motifs import FrequencyBias
from pysgg.modeling.roi_heads.relation_head.sampling import RelationSampling
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier
from pysgg.modeling.roi_heads.relation_head.model_bgnn import (
    LearnableRelatednessGating,
    MessagePassingUnitGatingWithRelnessLogits,
    MessagePassingUnit_v1,
    MessagePassingUnit_v2,
    MessageFusion
)
from util.dataset import get_dataset_statistics
from transformers import BertTokenizer, BertModel
import json
# 初始化BERT的tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
# 字典文件路径
dict_file = "/home/p_zhuzy/p_zhu/PySGG-main/datasets/vg/VG-SGG-dicts-with-attri.json"

# 加载 JSON 文件
with open(dict_file, 'r') as f:
    data = json.load(f)
idx_to_predicate = data["idx_to_predicate"]

class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, args, in_channels, half_out=False, cat_all_levels=False,for_relation = False):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = 7
        scales = [0.25,0.125,0.0625,0.03125]
        sampling_ratio = 2
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        input_size = in_channels * resolution ** 2
        representation_size = 4096
        use_gn = False
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)

        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size
        
        self.fc7 = make_fc(representation_size, out_dim, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

class FPNPredictor(nn.Module):
    def __init__(self, args, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = args.box_head_num_class
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes =  num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_logit, bbox_pred

#########Box Head#####
def make_roi_box_post_processor(ags):
    use_fpn = True

    bbox_reg_weights = [10.0,10.0,5.0,5.0]
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = 0.01
    nms_thresh = 0.3
    detections_per_img = 80
    cls_agnostic_bbox_reg = False
    bbox_aug_enabled = False
    post_nms_per_cls_topn = 300
    nms_filter_duplicates = True
    save_proposals = False

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        post_nms_per_cls_topn,
        nms_filter_duplicates,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled,
        save_proposals
    )
    return postprocessor

def make_roi_box_samp_processor(args):
    matcher = Matcher(
        0.5,0.3,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = [10.0,10.0,5.0,5.0]
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(256, 0.5)

    samp_processor = FastRCNNSampling(
        matcher,
        fg_bg_sampler,
        box_coder,
    )

    return samp_processor

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, args, in_channels):
        super(ROIBoxHead, self).__init__()
        self.args = args
        #进行特征提取
        self.feature_extractor = FPN2MLPFeatureExtractor(args, in_channels,
                                                                half_out=False,
                                                                for_relation=False)
        self.predictor = FPNPredictor(
            args, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(args)
        self.loss_evaluator = FastRCNNLossComputation(False)
        self.samp_processor = make_roi_box_samp_processor(args)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        ###################################################################
        # box head specifically for relation prediction model
        ###################################################################
        # if self.cfg.MODEL.RELATION_ON:
        #     if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        #         # use ground truth box as proposals
        #         proposals = [target.copy_with_fields(["labels", "attributes"]) for target in targets]
        #         x = self.feature_extractor(features, proposals)
        #         if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
        #             # mode==predcls
        #             # return gt proposals and no loss even during training
        #             return x, proposals, {}
        #         else:
        #             # mode==sgcls
        #             # add field:class_logits into gt proposals, note field:labels is still gt
        #             class_logits, _ = self.predictor(x)
        #             proposals = add_predict_info(proposals, class_logits)
        #             return x, proposals, {}
        #     else:
                # mode==sgdet
                # add the instance labels for the following instances classification on refined features
        if self.training:
            assert targets is not None
            proposals = self.samp_processor.assign_label_to_proposals(proposals, targets)
        x = self.feature_extractor(features, proposals)
        class_logits, box_regression = self.predictor(x)
        proposals = add_predict_logits(proposals, class_logits)
        # post process:
        # filter proposals using nms, keep original bbox, add a field 'boxes_per_cls' of size (#nms, #cls, 4)
        x, result = self.post_processor((x, class_logits, box_regression), proposals, relation_mode=True)
        # note x is not matched with processed_proposals, so sharing x is not permitted
        return x, result, {}

        # #####################################################################
        # # Original box head (relation_on = False)
        # #####################################################################
        # if self.training:
        #     # Faster R-CNN subsamples during training the proposals with a fixed
        #     # positive / negative ratio
        #     with torch.no_grad():
        #         proposals = self.samp_processor.subsample(proposals, targets)

        # # extract features that will be fed to the final classifier. The
        # # feature_extractor generally corresponds to the pooler + heads
        # x = self.feature_extractor(features, proposals)
        # # final classifier that converts the features into predictions
        # class_logits, box_regression = self.predictor(x)
        # proposals = add_predict_logits(proposals, class_logits)

        # if not self.training:
        #     x, result = self.post_processor((x, class_logits, box_regression), proposals)

        #     # if we want to save the proposals, we need sort them by confidence first.
        #     if self.cfg.TEST.SAVE_PROPOSALS:
        #         _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
        #         x = x[sort_ind]
        #         result = result[sort_ind]
        #         result.add_field("features", x.cpu().numpy())

        #     return x, result, {}

        # loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], proposals)

        # return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

 
#########Relation Head#####
class RelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, args, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        # self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = 7
        pool_all_levels = True
        self.feature_adapter = nn.Linear(4864, 4096)
        
        self.feature_extractor = FPN2MLPFeatureExtractor(args, in_channels, cat_all_levels=pool_all_levels)
        self.out_channels = self.feature_extractor.out_channels

        # separete spatial
        self.separate_spatial = False
        if self.separate_spatial: #false
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim // 2), nn.ReLU(inplace=True),
                                                make_fc(
                                                    out_dim // 2, out_dim), nn.ReLU(inplace=True),
                                                ])

        # union rectangle size 空间矩形处理部分
        self.rect_size = resolution * 4 -1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            ])
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_relation_bert_embeddings(self, relation_labels):
    # 将关系类别的文本标签传递给tokenizer进行编码
        encoding = self.tokenizer(relation_labels, padding=True, truncation=True, return_tensors='pt', max_length=32)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 获取 BERT 的输出
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            
        # 使用 [CLS] token 作为句子的嵌入表示
        relation_bert_embeddings = outputs.last_hidden_state[:, 0, :]
    
        return relation_bert_embeddings

    def forward(self, x, proposals, rel_pair_idxs=None):
        device = x[0].device
        union_proposals = [] #存储联合区域
        rect_inputs = [] #存储空间矩形输入
        relation_labels = []
        ## 遍历每个图像的proposal和关系对
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            ## 获取头尾对象的proposal
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            ## 计算联合区域（包含头尾的最小矩形）
            union_proposal = boxlist_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

            # use range to construct rectangle, sized (rect_size, rect_size)
            #构建空间矩形特征
            num_rel = len(rel_pair_idx) ## 当前图像的关系对数量
            # # 创建27x27的坐标网格
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
            # resize bbox to the scale rect_size
            ## 调整proposal到27x27尺度
            head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
            tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
            ## 生成头部和尾部矩形掩膜（bool转float）
            head_rect = ((dummy_x_range >= head_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= head_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= head_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= head_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= tail_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= tail_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= tail_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()

            ## 拼接两个掩膜作为输入（通道维度）
            rect_input = torch.stack((head_rect, tail_rect), dim=1) # (num_rel, 4, rect_size, rect_size)
            rect_inputs.append(rect_input)

                # 收集所有图像的关系标签
            for pair_idx in rel_pair_idx:
                head_idx, tail_idx = pair_idx
                relation_label = idx_to_predicate.get(head_idx, 'unknown')
                relation_labels.append(relation_label)  # 添加到全局列表

        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        #处理所有图像的输入
        rect_inputs = torch.cat(rect_inputs, dim=0)
        rel_bert_embeddings = self.get_relation_bert_embeddings(relation_labels)
        #对应公式中的f_p(e_i ⊕ e_j)
        rect_features = self.rect_conv(rect_inputs) ## 通过卷积网络提取空间特征

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        #对应公式中的f_u(up_{i,j})，提取联合区域视觉特征
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)
        # merge two parts
        if self.separate_spatial: #false
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else: 
            union_features = union_vis_features + rect_features # 特征相加
            union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels)
        #print("un",union_features.shape)
        #print("rel_bert_embeddings",rel_bert_embeddings.shape)

        union_features = torch.cat((union_features, rel_bert_embeddings), dim=1)
        #print('union_features',union_features.shape)
        union_features = self.feature_adapter(union_features)  # [1390,4864]->[1390,4096]
        # 对应公式中的f_r(r_ij)
        #print(f"拼接后形状: {union_features.shape}") #[750, 2048]
        return union_features #返回最终特征 [total_rel, out_dim]
        

class PairwiseFeatureExtractor(nn.Module):
    """
    extract the pairwise features from the object pairs and union features.
    most pipeline keep same with the motifs instead the lstm massage passing process
    """

    def __init__(self, args, in_channels):
        super(PairwiseFeatureExtractor, self).__init__()
        # self.cfg = config
        statistics = get_dataset_statistics(args)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes

        # mode
        self.mode = 'sgdet'

        # features augmentation for instance features
        # word embedding
        # add language prior representation according to the prediction distribution
        # of objects
        self.embed_dim = 768
        #self.embed_dim = 300
        self.obj_dim = in_channels
        self.hidden_dim = 512
        self.pooling_dim = 2048

        self.word_embed_feats_on = True
        if self.word_embed_feats_on:
            #obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=args.glove_dir, wv_dim=self.embed_dim)
            #obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=args.word2vec_dir, wv_dim=self.embed_dim)
            obj_embed_vecs = obj_edge_vectors_bert(self.obj_classes, wv_dir=args.bert_dir, wv_dim=self.embed_dim)
            self.obj_embed_on_prob_dist = nn.Embedding(self.num_obj_classes, self.embed_dim)
            self.obj_embed_on_pred_label = nn.Embedding(self.num_obj_classes, self.embed_dim)
            with torch.no_grad():
                self.obj_embed_on_prob_dist.weight.copy_(obj_embed_vecs, non_blocking=True)
                self.obj_embed_on_pred_label.weight.copy_(obj_embed_vecs, non_blocking=True)
        else:
            self.embed_dim = 0

        # features augmentation for rel pairwise features
        self.rel_feature_type = 'fusion'

        # the input dimension is ROI head MLP, but the inner module is pooling dim, so we need
        # to decrease the dimension first.
        if self.pooling_dim != in_channels:
            self.rel_feat_dim_not_match = True
            self.rel_feature_up_dim = make_fc(in_channels, self.pooling_dim)
            layer_init(self.rel_feature_up_dim, xavier=True)
        else:
            self.rel_feat_dim_not_match = False

        self.pairwise_obj_feat_updim_fc = make_fc(self.hidden_dim + self.obj_dim + self.embed_dim,
                                                  self.hidden_dim * 2)

        self.outdim = self.pooling_dim
        # position embedding
        # encode the geometry information of bbox in relationships
        self.geometry_feat_dim = 128
        self.pos_embed = nn.Sequential(*[
            make_fc(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            make_fc(32, self.geometry_feat_dim), nn.ReLU(inplace=True),
        ])

        if self.rel_feature_type in ["obj_pair", "fusion"]:
            self.spatial_for_vision = True
            if self.spatial_for_vision:
                self.spt_emb = nn.Sequential(*[make_fc(32, self.hidden_dim),
                                               nn.ReLU(inplace=True),
                                               make_fc(self.hidden_dim, self.hidden_dim * 2),
                                               nn.ReLU(inplace=True)
                                               ])
                layer_init(self.spt_emb[0], xavier=True)
                layer_init(self.spt_emb[2], xavier=True)

            self.pairwise_rel_feat_finalize_fc = nn.Sequential(
                make_fc(self.hidden_dim * 2, self.pooling_dim),
                nn.ReLU(inplace=True),
            )

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.obj_hidden_linear = make_fc(self.obj_dim + self.embed_dim + self.geometry_feat_dim, self.hidden_dim)

        self.obj_feat_aug_finalize_fc = nn.Sequential(
            make_fc(self.hidden_dim + self.obj_dim + self.embed_dim, self.pooling_dim),
            nn.ReLU(inplace=True),
        )

        # untreated average features

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder
    #关系特征生成函数：
    def pairwise_rel_features(self, augment_obj_feat, union_features, rel_pair_idxs, inst_proposals):
        #对每个图像的边界框提案进行几何特征编码
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in inst_proposals]
        #获取每个图像中的对象数量。
        num_objs = [len(p) for p in inst_proposals]
        # post decode
        # (num_objs, hidden_dim) -> (num_objs, hidden_dim * 2)
        # going to split single object representation to sub-object role of relationship
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(pairwise_obj_feats_fused.size(0), 2, self.hidden_dim)
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)
        # split分离头实体和尾实体特征
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        # generate the pairwise object for relationship representation
        # (num_objs, hidden_dim) <rel pairing > (num_objs, hidden_dim)
        #   -> (num_rel, hidden_dim * 2)
        #   -> (num_rel, hidden_dim)
        #收集关系对特征和边界框几何信息。
        obj_pair_feat4rel_rep = []
        pair_bboxs_info = []
        #遍历图像构建关系对特征。
        for pair_idx, head_rep, tail_rep, obj_box in zip(rel_pair_idxs, head_reps, tail_reps, obj_boxs):
            obj_pair_feat4rel_rep.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        pair_bbox_geo_info = cat(pair_bboxs_info, dim=0)
        obj_pair_feat4rel_rep = cat(obj_pair_feat4rel_rep, dim=0)  # (num_rel, hidden_dim * 2)
        if self.spatial_for_vision:
            obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * self.spt_emb(pair_bbox_geo_info)

        obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(obj_pair_feat4rel_rep)  # (num_rel, hidden_dim)

        return obj_pair_feat4rel_rep

    def forward(self, inst_roi_feats, union_features, inst_proposals, rel_pair_idxs, ):
        """

        :param inst_roi_feats: instance ROI features, list(Tensor)
        :param inst_proposals: instance proposals, list(BoxList())
        :param rel_pair_idxs:
        :return:
            obj_pred_logits obj_pred_labels 2nd time instance classification results
            obj_representation4rel, the objects features ready for the represent the relationship
        """
        # using label or logits do the label space embeddings
        if self.training or False:#从提案中获取真实对象标签（labels字段）
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
        else:
            obj_labels = None

        if self.word_embed_feats_on:#true 语义嵌入处理：将对象类别预测转化为语义向量
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in inst_proposals], dim=0).detach()
            obj_embed_by_pred_dist = F.softmax(obj_logits, dim=1) @ self.obj_embed_on_prob_dist.weight

        # box positive geometry embedding
        assert inst_proposals[0].mode == 'xyxy'
        #几何特征：通过 MLP 将几何信息映射到 128 维空间
        pos_embed = self.pos_embed(encode_box_info(inst_proposals))

        # word embedding refine
        batch_size = inst_roi_feats.shape[0]
        if self.word_embed_feats_on:#三种特征的拼接
            obj_pre_rep = cat((inst_roi_feats, obj_embed_by_pred_dist, pos_embed), -1)
        else:
            obj_pre_rep = cat((inst_roi_feats, pos_embed), -1)
        # object level contextual feature
        #得到对象特征增强后的表示
        augment_obj_feat = self.obj_hidden_linear(obj_pre_rep)  # map to hidden_dim

        # todo reclassify on the fused object features
        # Decode in order
        if self.mode != 'predcls':
            # todo: currently no redo classification on embedding representation,
            #       we just use the first stage object prediction
            obj_pred_labels = cat([each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0)
        else: #sgdet:使用检测器的预测标签作为对象标签
            assert obj_labels is not None
            obj_pred_labels = obj_labels#obj_pred_labels 直接赋值为 obj_labels，表示对象的预测标签。

        # object labels space embedding from the prediction labels
        if self.word_embed_feats_on:#直接通过预测标签索引词嵌入矩阵
            obj_embed_by_pred_labels = self.obj_embed_on_pred_label(obj_pred_labels.long())

        # average action in test phrase for causal effect analysis
        if self.word_embed_feats_on:#最终对象特征拼接
            augment_obj_feat = cat((obj_embed_by_pred_labels, inst_roi_feats, augment_obj_feat), -1)
        else:
            augment_obj_feat = cat((inst_roi_feats, augment_obj_feat), -1)
        #rel_feature_type = "fusion"
        #关系特征生成
        if self.rel_feature_type == "obj_pair" or self.rel_feature_type == "fusion":
            #计算对象对（pairwise objects）的关系特征
            rel_features = self.pairwise_rel_features(augment_obj_feat, union_features,
                                                      rel_pair_idxs, inst_proposals)
            if self.rel_feature_type == "fusion":
                if self.rel_feat_dim_not_match:#维度调整
                    union_features = self.rel_feature_up_dim(union_features)
                #谓词初始表示的公式
                rel_features = union_features + rel_features

        elif self.rel_feature_type == "union":
            if self.rel_feat_dim_not_match:
                union_features = self.rel_feature_up_dim(union_features)
            rel_features = union_features

        else:
            assert False
        # mapping to hidden
        augment_obj_feat = self.obj_feat_aug_finalize_fc(augment_obj_feat)
        #返回 最终增强的对象特征 和 关系特征
        return augment_obj_feat, rel_features
    
class CrossModalFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim  # 保存为成员变量
        self.vis_query = nn.Linear(hidden_dim, hidden_dim)
        self.symb_key = nn.Linear(hidden_dim, hidden_dim)
        self.symb_value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vis_feat, symb_feat):
        Q = self.vis_query(vis_feat)  # [N, D]
        K = self.symb_key(symb_feat)  # [N, D]
        V = self.symb_value(symb_feat)
        
        # 使用self.hidden_dim或动态计算scale
        attn_scores = Q @ K.transpose(-2,-1) / torch.sqrt(torch.tensor(self.hidden_dim, device=Q.device))
        attn = torch.softmax(attn_scores, dim=-1)
        return attn @ V  # [N, D]

#得到关系分类得分和相关性得分；
# RelAwareRelFeature 类用于计算图像中物体之间的关系特征。
# forward 方法整合了物体的位置信息、语义信息和视觉特征，
# 将这些信息组合后，通过神经网络进行预测并输出物体对之间的关系强度。
# 这种结构能够有效地捕捉到物体与物体之间的关系，为后续的关系推理提供支持。
class RelAwareRelFeature(nn.Module):
    def __init__(
        self,
        args,
        input_dim,
    ):
        super(RelAwareRelFeature, self).__init__()
        self.args = args
        #关系类别数量，51个类别
        self.num_rel_cls = 51

        self.input_dim = input_dim

        self.predictor_type = ("hybrid")
        #物体类别数量为151
        self.num_obj_classes = 151
        self.embed_dim = 768
        #self.embed_dim = 300
        self.geometry_feat_dim = 128
        self.roi_feat_dim = 4096
        self.hidden_dim = 512
        #获取数据集的统计信息
        statistics = get_dataset_statistics(args)
        obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]

        # obj_embed_vecs = obj_edge_vectors(
        #     obj_classes, wv_dir=args.glove_dir, wv_dim=self.embed_dim
        # )
        # obj_embed_vecs = obj_edge_vectors(
        #     obj_classes, wv_dir=args.word2vec_dir, wv_dim=self.embed_dim
        # )
        #物体的语义嵌入
        obj_embed_vecs = obj_edge_vectors_bert(
            obj_classes, wv_dir=args.bert_dir, wv_dim=self.embed_dim
        )
        self.obj_sem_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)

        with torch.no_grad():
            self.obj_sem_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
        #物体的位置几何嵌入
        self.obj_pos_embed = nn.Sequential(
            nn.Linear(9, self.geometry_feat_dim),
            nn.ReLU(),
            nn.Linear(self.geometry_feat_dim, self.geometry_feat_dim),
        )
        #定义是否使用视觉特征
        self.visual_features_on = (True)

        self.proposal_box_feat_extract = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                self.embed_dim * 2 + self.geometry_feat_dim * 2,
                self.hidden_dim,
            ),
        )

        if self.visual_features_on:
            self.vis_embed = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.input_dim, self.hidden_dim),
            )

            self.proposal_feat_fusion = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
            self.cross_att = CrossModalFusion(self.hidden_dim)

            

        self.out_dim = self.num_rel_cls - 1
        #关系分类器
        self.proposal_relness_cls_fc = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        if self.predictor_type == "hybrid":
            self.fusion_layer = nn.Linear(self.out_dim, 1)

    def forward(
        self,
        visual_feat, #输入的视觉特征
        entities_proposals, #实体的proposal
        rel_pair_inds, #关系对的索引
    ):

        relness_matrix = [] #存储关系强度矩阵
        relness_logits_batch = [] #存储关系分类的logits
        #处理视觉图为true
        if self.visual_features_on:
            visual_feat = self.vis_embed(visual_feat.detach())
        #拆分视觉特征，根据关系对的索引分割成对的视觉特征
        visual_feat_split = torch.split(visual_feat, [len(p) for p in rel_pair_inds], dim=0)
        #依次遍历每个提案：遍历每张图像的提案，包括相关的视觉特征和关系对。
        for img_id, (proposal, vis_feats, pair_idx) in enumerate(
            zip(entities_proposals, visual_feat_split, rel_pair_inds)
        ):
            #获取预测 logits
            pred_logits = proposal.get_field("predict_logits").detach()
            device = proposal.bbox.device
            #初始话关系矩阵为零矩阵，用于存储每个提案之间的关系评分
            pred_rel_matrix = torch.zeros(
                (len(proposal), len(proposal)), device=device, dtype=pred_logits.dtype
            )
            #获取位置嵌入
            pos_embed = self.obj_pos_embed(
                encode_box_info(
                    [
                        proposal,
                    ]
                )
            )
            #计算语义嵌入
            obj_sem_embed = F.softmax(pred_logits, dim=1) @ self.obj_sem_embed.weight
            #根据索引组合两个物体的位置信息和语义特征，形成关系对的特征表示
            rel_pair_symb_repre = torch.cat( 
                (
                    pos_embed[pair_idx[:, 0]],#p_i 的几何特征
                    obj_sem_embed[pair_idx[:, 0]], #p_i 的语义特征
                    pos_embed[pair_idx[:, 1]], #p_j 的几何特征
                    obj_sem_embed[pair_idx[:, 1]], #p_j 的语义特征
                ),
                dim=1,
            )
            #提取关系对特征：通过 self.proposal_box_feat_extract 处理组合特征。
            prop_pair_geo_feat = self.proposal_box_feat_extract(rel_pair_symb_repre)
            #如果启用视觉特征，将视觉特征与关系特征结合，反之则只使用关系特征。
            if self.visual_features_on:#true
                # visual_relness_feat = self.self_att(vis_feats, vis_feats, vis_feats).squeeze(1)
                visual_relness_feat = vis_feats #视觉特征 r_{i→j}
                # rel_prop_repre = self.proposal_feat_fusion(
                #     torch.cat((visual_relness_feat, prop_pair_geo_feat), dim=1)
                # )
                rel_prop_repre = self.cross_att(visual_relness_feat, prop_pair_geo_feat)

            else:
                rel_prop_repre = prop_pair_geo_feat
            #计算关系强度得分
            #relness_logits对应公式里的s，表示每个关系对在不同关系类别上的得分
            relness_logits = self.proposal_relness_cls_fc(rel_prop_repre)
            relness_logits = squeeze_tensor(relness_logits)
            #根据预测类型，计算关系强度分数（relness_scores），并将其存入预测的关系矩阵中。
            # 如果是混合类型，还会对 logits 进行合并处理
            if self.predictor_type == "hybrid": #是混合模型
                relness_bin_logits = self.fusion_layer(relness_logits)
                #sb是relness_scores
                relness_scores = squeeze_tensor(torch.sigmoid(relness_bin_logits))
                pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness_scores

                relness_logits = torch.cat((relness_logits, relness_bin_logits), dim=1)
            elif self.predictor_type == "single":
                relness_scores = squeeze_tensor(torch.sigmoid(relness_logits))
                pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness_scores.max(dim=1)[0]
            #存储每个图像的关系 logits 和矩阵：
            # 将本次计算得到的 logits 和关系矩阵分别加入到输出列表中。
            relness_logits_batch.append(relness_logits)

            relness_matrix.append(pred_rel_matrix)

        return (
            torch.cat(relness_logits_batch),
            relness_matrix,
        )

class BGNNContext(nn.Module):
    def __init__(
        self,
        args,
        in_channels,
        hidden_dim=1024,
        num_iter=2, #图神经网络的迭代次数为2
        dropout=False,
        gate_width=128,
        use_kernel_function=False,
    ):
        super(BGNNContext, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.update_step = num_iter

        if self.update_step < 1:
            print(
                "WARNING: the update_step should be greater than 0, current: ",
                +self.update_step,
            )
        #PairwiseFeatureExtractor返回 最终增强的对象特征公式1 和 关系特征公式2
        self.pairwise_feature_extractor = PairwiseFeatureExtractor(args, in_channels)
        self.pooling_dim = self.pairwise_feature_extractor.pooling_dim
        #定义关系感知模块
        self.rel_aware_on = True
        self.rel_aware_module_type = (RelAwareRelFeature)
        #定义关系分类的数量
        self.num_rel_cls = 51
        #初始化关系感知的一些标志变量
        self.relness_weighting_mp = False
        self.gating_with_relness_logits = False
        self.filter_the_mp_instance = False
        self.relation_conf_aware_models = None
        self.apply_gt_for_rel_conf = False
        #初始化消息传递迭代次数、图过滤方法和有效对数目。
        self.mp_pair_refine_iter = 1

        self.graph_filtering_method = ("RelAwareRelFeature")

        self.vail_pair_num = 128
        #关系感知模块构建
        if self.rel_aware_on: #true

            #####  build up the relationship aware modules
            self.mp_pair_refine_iter = (3)
            assert self.mp_pair_refine_iter > 0
            #定义是否共享预训练的关系分类器
            self.shared_pre_rel_classifier = (False)

            if self.mp_pair_refine_iter <= 1:
                self.shared_pre_rel_classifier = False

            if not self.shared_pre_rel_classifier:
                self.relation_conf_aware_models = nn.ModuleList()
                for ii in range(self.mp_pair_refine_iter):

                    if ii == 0:
                        input_dim = self.pooling_dim
                    else:
                        input_dim = self.hidden_dim
                    self.relation_conf_aware_models.append(
                        RelAwareRelFeature(args,input_dim)
                    )
            else: #共享分类器，直接实例化一个关系感知模型
                input_dim = self.pooling_dim
                self.relation_conf_aware_models = RelAwareRelFeature(args,input_dim)
            self.pretrain_pre_clser_mode = False

            ######  relationship confidence recalibration
            #关系置信度重新标定
            self.apply_gt_for_rel_conf = False

            self.gating_with_relness_logits = (False)
            self.relness_weighting_mp = (True)
            # 'minmax',  'learnable_scaling'
            #设置关系得分重标定的方法。
            self.relness_score_recalibration_method = ("learnable_scaling")
            #阿尔法=3.12 贝塔=0.06
            if self.relness_score_recalibration_method == "learnable_scaling":
                self.learnable_relness_score_gating_recalibration = (
                    LearnableRelatednessGating()
                )
            elif self.relness_score_recalibration_method == "minmax":
                self.min_relness = nn.Parameter(
                    torch.Tensor(
                        [
                            1e-5,
                        ]
                    ),
                    requires_grad=False,
                )
                self.max_relness = nn.Parameter(
                    torch.Tensor(
                        [
                            0.5,
                        ]
                    ),
                    requires_grad=False,
                )
            else:
                raise ValueError(
                    "Invalid relness_score_recalibration_method "
                    + self.relness_score_recalibration_method
                )

            self.filter_the_mp_instance = (True)

        # decrease the dimension before mp
        #消息传递前的维度调整
        #创建对象特征和关系特征的降维全连接层
        self.obj_downdim_fc = nn.Sequential(
            make_fc(self.pooling_dim, self.hidden_dim),
            nn.ReLU(True),
        )
        self.rel_downdim_fc = nn.Sequential(
            make_fc(self.pooling_dim, self.hidden_dim),
            nn.ReLU(True),
        )
        #创建一个用于融合成对对象特征和关系特征的序列网络
        self.obj_pair2rel_fuse = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * 2),
            make_fc(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
        )
        #定义一个用于填充的特征，初始化为零向量
        self.padding_feature = nn.Parameter(
            torch.zeros((self.hidden_dim)), requires_grad=False
        )
        #消息传递单元的选择与定义
        #根据参数选择使用的消息传递单元
        if use_kernel_function: #use_kernel_function=False
            MessagePassingUnit = MessagePassingUnit_v2
        else:#用v1的消息传递单元
            MessagePassingUnit = MessagePassingUnit_v1
        #设置参数共享选项
        self.share_parameters_each_iter = (False)
        #设置参数集数量
        param_set_num = num_iter
        if self.share_parameters_each_iter:
            param_set_num = 1
        
        #门控机制定义
        #使用消息传递单元定义不同方向的门控机制
        self.gate_sub2pred = nn.Sequential(
            *[MessagePassingUnit(self.hidden_dim, gate_width) for _ in range(param_set_num)]
        )
        self.gate_obj2pred = nn.Sequential(
            *[MessagePassingUnit(self.hidden_dim, gate_width) for _ in range(param_set_num)]
        )
        self.gate_pred2sub = nn.Sequential(
            *[MessagePassingUnit(self.hidden_dim, gate_width) for _ in range(param_set_num)]
        )
        self.gate_pred2obj = nn.Sequential(
            *[MessagePassingUnit(self.hidden_dim, gate_width) for _ in range(param_set_num)]
        )
        #如果启用与关系得分相关的门控功能，则改变消息传递单元的定义。
        if self.gating_with_relness_logits:#false
            MessagePassingUnit = MessagePassingUnitGatingWithRelnessLogits
            self.gate_pred2sub = nn.Sequential(
                *[
                    MessagePassingUnit(
                        self.hidden_dim, self.num_rel_cls, self.relness_weighting_mp
                    )
                    for _ in range(param_set_num)
                ]
            )
            self.gate_pred2obj = nn.Sequential(
                *[
                    MessagePassingUnit(
                        self.hidden_dim, self.num_rel_cls, self.relness_weighting_mp
                    )
                    for _ in range(param_set_num)
                ]
            )
        #定义针对对象和预测消息的融合网络
        self.object_msg_fusion = nn.Sequential(
            *[MessageFusion(self.hidden_dim, dropout) for _ in range(param_set_num)]
        )  #
        self.pred_msg_fusion = nn.Sequential(
            *[MessageFusion(self.hidden_dim, dropout) for _ in range(param_set_num)]
        )
        #定义输出跳接连接的标志，并初始化向前传播计时器
        self.output_skip_connection = (False)

        self.forward_time = 0
    #设置预训练模式
    def set_pretrain_pre_clser_mode(self, val=True):
        self.pretrain_pre_clser_mode = val
    #定义规范化关系得分的方法
    def normalize(self, each_img_relness, selected_rel_prop_pairs_idx):
        #计算当前的最大和最小关系得分
        if len(squeeze_tensor(torch.nonzero(each_img_relness != 1.0))) > 10:
            select_relness_for_minmax = each_img_relness[selected_rel_prop_pairs_idx]
            curr_relness_max = select_relness_for_minmax.detach()[
                int(len(select_relness_for_minmax) * 0.05) :
            ].max()
            curr_relness_min = select_relness_for_minmax.detach().min()
            #用加权方式计算新的最小和最大值
            min_val = self.min_relness.data * 0.7 + curr_relness_min * 0.3
            max_val = self.max_relness.data * 0.7 + curr_relness_max * 0.3

            #在训练过程中使用移动平均来更新最小和最大值
            if self.training:
                # moving average for the relness scores normalization
                self.min_relness.data = self.min_relness.data * 0.9 + curr_relness_min * 0.1
                self.max_relness.data = self.max_relness.data * 0.9 + curr_relness_max * 0.1

        else:#如果没有足够的值，则使用默认的最小和最大值
            min_val = self.min_relness
            max_val = self.max_relness
        #定义一个用于进行最小-最大规范化的辅助函数
        def minmax_norm(data, min, max):
            return (data - min) / (max - min + 1e-5)
        #对于所有非1.0的关系得分进行规范化，并确保其值在[0, 1]之间。
        # apply on all non 1.0 relness scores
        each_img_relness[each_img_relness != 1.0] = torch.clamp(
            minmax_norm(each_img_relness[each_img_relness != 1.0], min_val, max_val),
            max=1.0,
            min=0.0,
        )
        #返回规范化后的关系得分
        return each_img_relness
    #排名最小-最大重标定，定义了一种重标定关系得分的方法
    def ranking_minmax_recalibration(self, each_img_relness, selected_rel_prop_pairs_idx):

        # normalize the relness score
        #调用规范化方法来对关系得分进行规范化
        each_img_relness = self.normalize(each_img_relness, selected_rel_prop_pairs_idx)
        #计算所选关系对的总数
        # take the top 10% pairs set as the must keep relationship by set it relness into 1.0
        total_rel_num = len(selected_rel_prop_pairs_idx)
        #将前10%的关系得分设置为1.0，以保持这些关系
        each_img_relness[selected_rel_prop_pairs_idx[: int(total_rel_num * 0.1)]] += (
            1.0
            - each_img_relness[selected_rel_prop_pairs_idx[: int(total_rel_num * 0.1)]]
        )
        #返回重新标定后的关系得分
        return each_img_relness
    #关系得分重标定，根据设置的重标定方法，对关系得分进行重标定后返回；就是公式7里的T
    def relness_score_recalibration(self, each_img_relness, selected_rel_prop_pairs_idx):
        if self.relness_score_recalibration_method == "minmax":#flase
            each_img_relness = self.ranking_minmax_recalibration(
                each_img_relness, selected_rel_prop_pairs_idx
            )
        elif self.relness_score_recalibration_method == "learnable_scaling":#true
            #可学习的缩放（对应 α 和 β）
            each_img_relness = self.learnable_relness_score_gating_recalibration(
                each_img_relness
            )
        return each_img_relness
    #准备邻接矩阵
    def _prepare_adjacency_matrix(self, proposals, rel_pair_idxs, relatedness):
        """
        prepare the index of how subject and object related to the union boxes
        :param num_proposals:
        :param rel_pair_idxs:
        :return:c
            ALL RETURN THINGS ARE BATCH-WISE CONCATENATED

            rel_inds,
                extent the instances pairing matrix to the batch wised (num_rel, 2)
            subj_pred_map,
                how the instances related to the relation predicates as the subject (num_inst, rel_pair_num)
            obj_pred_map
                how the instances related to the relation predicates as the object (num_inst, rel_pair_num)
            selected_relness,
                the relatness score for selected relationship proposal that send message to adjency nodes (val_rel_pair_num, 1)
            selected_rel_prop_pairs_idx
                the relationship proposal id that selected relationship proposal that send message to adjency nodes (val_rel_pair_num, 1)
        """
        #初始化存储关系对的索引和其他相关变量
        rel_inds_batch_cat = []
        offset = 0
        num_proposals = [len(props) for props in proposals]
        rel_prop_pairs_relness_batch = []
        #遍历 proposals 和 rel_pair_idxs，处理每张图片的对象及其关系对。
        for idx, (prop, rel_ind_i) in enumerate(
            zip(
                proposals,
                rel_pair_idxs,
            )
        ):
            #如果需要筛选实例，则获取每个关系对的相关性得分
            if self.filter_the_mp_instance: #true
                assert relatedness is not None
                related_matrix = relatedness[idx]
                rel_prop_pairs_relness = related_matrix[rel_ind_i[:, 0], rel_ind_i[:, 1]]

                det_score = prop.get_field("pred_scores")

                rel_prop_pairs_relness_batch.append(rel_prop_pairs_relness)
            #复制索引，添加偏移量，然后保存组合索引
            rel_ind_i = copy.deepcopy(rel_ind_i)

            rel_ind_i += offset
            offset += len(prop)
            rel_inds_batch_cat.append(rel_ind_i)
        #合并所有索引
        rel_inds_batch_cat = torch.cat(rel_inds_batch_cat, 0)
        # 两个矩阵指示对象是否作为关系对的主体或客体
        #初始化一个矩阵，用于存储主体与关系对的映射
        subj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )
        #同样初始化一个矩阵，存储客体与关系对的映射
        obj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )
        # only message passing on valid pairs
        #有效对的消息传递
        #如果有有效的关系对，则继续处理
        if len(rel_prop_pairs_relness_batch) != 0:

            if self.rel_aware_on:#true
                offset = 0
                rel_prop_pairs_relness_sorted_idx = []
                rel_prop_pairs_relness_batch_update = []
                #遍历关系对的相关性得分
                for idx, each_img_relness in enumerate(rel_prop_pairs_relness_batch):

                    (
                        #对关系得分进行排序，选择分数最高的关系对
                        selected_rel_prop_pairs_relness,
                        selected_rel_prop_pairs_idx,
                    ) = torch.sort(each_img_relness, descending=True)
                    #根据情况选择要保留的关系对
                    if self.apply_gt_for_rel_conf:#false
                        # add the non-GT rel pair dynamically according to the GT rel num
                        gt_rel_idx = squeeze_tensor(
                            torch.nonzero(selected_rel_prop_pairs_relness == 1.0)
                        )
                        pred_rel_idx = squeeze_tensor(
                            torch.nonzero(selected_rel_prop_pairs_relness < 1.0)
                        )
                        pred_rel_num = int(len(gt_rel_idx) * 0.2)
                        pred_rel_num = (
                            pred_rel_num
                            if pred_rel_num < len(pred_rel_idx)
                            else len(pred_rel_idx)
                        )
                        pred_rel_num = pred_rel_num if pred_rel_num > 0 else 5
                        selected_rel_prop_pairs_idx = torch.cat(
                            (
                                selected_rel_prop_pairs_idx[gt_rel_idx],
                                selected_rel_prop_pairs_idx[pred_rel_idx[:pred_rel_num]],
                            )
                        )
                    else:
                        # recaliberating the relationship confidence for weighting
                        selected_rel_prop_pairs_idx = selected_rel_prop_pairs_idx[
                            : self.vail_pair_num
                        ]

                        if self.relness_weighting_mp and not self.pretrain_pre_clser_mode:
                            each_img_relness = self.relness_score_recalibration(
                                each_img_relness, selected_rel_prop_pairs_idx
                            )

                            selected_rel_prop_pairs_idx = squeeze_tensor(torch.nonzero(each_img_relness > 0.0001))

                    rel_prop_pairs_relness_batch_update.append(each_img_relness)

                    rel_prop_pairs_relness_sorted_idx.append(
                        selected_rel_prop_pairs_idx + offset
                    )
                    offset += len(each_img_relness)

                selected_rel_prop_pairs_idx = torch.cat(rel_prop_pairs_relness_sorted_idx, 0)
                rel_prop_pairs_relness_batch_cat = torch.cat(
                    rel_prop_pairs_relness_batch_update, 0
                )
            #生成映射矩阵
            #更新主体和客体的映射矩阵
            subj_pred_map[
                rel_inds_batch_cat[selected_rel_prop_pairs_idx, 0],
                selected_rel_prop_pairs_idx,
            ] = 1
            obj_pred_map[
                rel_inds_batch_cat[selected_rel_prop_pairs_idx, 1],
                selected_rel_prop_pairs_idx,
            ] = 1
            selected_relness = rel_prop_pairs_relness_batch_cat
        else:
            # or all relationship pairs
            #如果没有有效关系对，默认操作，初始化映射矩阵
            selected_rel_prop_pairs_idx = torch.arange(
                len(rel_inds_batch_cat[:, 0]), device=rel_inds_batch_cat.device
            )
            selected_relness = None
            subj_pred_map.scatter_(0, (rel_inds_batch_cat[:, 0].contiguous().view(1, -1)), 1)
            obj_pred_map.scatter_(0, (rel_inds_batch_cat[:, 1].contiguous().view(1, -1)), 1)
        return (
            rel_inds_batch_cat,
            subj_pred_map,
            obj_pred_map,
            selected_relness,
            selected_rel_prop_pairs_idx,
        )

    # Here, we do all the operations out of loop, the loop is just to combine the features
    # Less kernel evoke frequency improve the speed of the model
    #准备消息传递
    def prepare_message(
        self,
        target_features,
        source_features,
        select_mat,
        gate_module,
        relness_scores=None,
        relness_logits=None,
    ):
        """
        generate the message from the source nodes for the following merge operations.

        Then the message passing process can be
        :param target_features: (num_inst, dim)
        :param source_features: (num_rel, dim)
        :param select_mat:  (num_inst, rel_pair_num)
        :param gate_module:
        :param relness_scores: (num_rel, )
        :param relness_logit (num_rel, num_rel_category)

        :return: messages representation: (num_inst, dim)
        """
        #定义用于生成消息的准备方法
        feature_data = []
        #初始化特征数据，如果选择矩阵的和为0，生成零张量
        if select_mat.sum() == 0:
            temp = torch.zeros(
                (target_features.size()[1:]),
                requires_grad=True,
                dtype=target_features.dtype,
                device=target_features.dtype,
            )
            feature_data = torch.stack(temp, 0)
        else:
            #如果选择矩阵不为零，从源特征和目标特征中选择相应的特征。
            transfer_list = (select_mat > 0).nonzero()#取非零元素的坐标
            source_indices = transfer_list[:, 1] #源节点坐标，即关系的索引
            target_indices = transfer_list[:, 0] #目标节点坐标，即物体的索引
            source_f = torch.index_select(source_features, 0, source_indices)#得到源节点特征，即关系特征
            target_f = torch.index_select(target_features, 0, target_indices)#得到目标节点特征，即物体特征

            #选择关系得分
            select_relness = relness_scores[source_indices]
            #使用门控功能生成消息
            if self.gating_with_relness_logits:#false
                assert relness_logits is not None

                # relness_dist =  relness_logits
                select_relness_dist = torch.sigmoid(relness_logits[source_indices])

                if self.relness_weighting_mp:
                    transferred_features, weighting_gate = gate_module(
                        target_f, source_f, select_relness_dist, select_relness
                    )
                else:
                    transferred_features, weighting_gate = gate_module(
                        target_f, source_f, select_relness_dist
                    )
            else:
                if self.relness_weighting_mp:#true
                    select_relness = relness_scores[transfer_list[:, 1]]
                    transferred_features, weighting_gate = gate_module(
                        target_f, source_f, select_relness
                    )
                else:
                    transferred_features, weighting_gate = gate_module(target_f, source_f)
            #构造全零聚合函数
            aggregator_matrix = torch.zeros(
                (target_features.shape[0], transferred_features.shape[0]),
                dtype=weighting_gate.dtype,
                device=weighting_gate.device,
            )
            #遍历所有物体节点
            for f_id in range(target_features.shape[0]):
                #判断物体是否参与关系
                if select_mat[f_id, :].data.sum() > 0:#检查物体f_id是否至少参与一个关系
                    # average from the multiple sources
                    #获取物体关联的关系索引
                    feature_indices = squeeze_tensor(
                        (transfer_list[:, 0] == f_id).nonzero()
                    )  # obtain source_relevant_idx
                    # (target, source_relevant_idx)
                    aggregator_matrix[f_id, feature_indices] = 1
            # (target, source_relevant_idx) @ (source_relevant_idx, feat-dim) => (target, feat-dim)
            aggregate_feat = torch.matmul(aggregator_matrix, transferred_features)
            #计算归一化因子
            avg_factor = aggregator_matrix.sum(dim=1)
            #标记有效物体
            vaild_aggregate_idx = avg_factor != 0
            #扩展归一化因子
            avg_factor = avg_factor.unsqueeze(1).expand(
                avg_factor.shape[0], aggregate_feat.shape[1]
            )
            #若某物体接收多条消息（如 A 同时接收 A→B 和 A→C），则会取平均值。
            aggregate_feat[vaild_aggregate_idx] /= avg_factor[vaild_aggregate_idx]

            feature_data = aggregate_feat
        return feature_data

    def pairwise_rel_features(self, augment_obj_feat, rel_pair_idxs):
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(
            pairwise_obj_feats_fused.size(0), 2, self.hidden_dim
        )
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)

        obj_pair_feat4rel_rep = torch.cat(
            (head_rep[rel_pair_idxs[:, 0]], tail_rep[rel_pair_idxs[:, 1]]), dim=-1
        )

        obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(
            obj_pair_feat4rel_rep
        )  # (num_rel, hidden_dim)

        return obj_pair_feat4rel_rep

    def forward(
        self,
        inst_features,#实例特征
        rel_union_features,#关系联合特征
        proposals,#rpn中得出的包含多个boxlist的列表
        rel_pair_inds,#关系对索引列表
        rel_gt_binarys=None,
        logger=None,
    ):
        """

        :param inst_features: instance_num, pooling_dim
        :param rel_union_features:  rel_num, pooling_dim
        :param proposals: instance proposals
        :param rel_pair_inds: relaion pair indices list(tensor)
        :param rel_binarys: [num_prop, num_prop] the relatedness of each pair of boxes
        :return:
        """
        #获取每个提案的实例数
        num_inst_proposals = [len(b) for b in proposals]
        #提取增强的对象特征和关系特征
        augment_obj_feat, rel_feats = self.pairwise_feature_extractor(
            inst_features,
            rel_union_features,
            proposals,
            rel_pair_inds,
        )

        relatedness_each_iters = []#存储每次迭代得出的相关性系数
        refine_rel_feats_each_iters = [rel_feats]#细化后的关系特征
        refine_ent_feats_each_iters = [augment_obj_feat]#细化后的实体特征
        pre_cls_logits_each_iter = []#存储预测的关系分类得分
        #迭代 mp_pair_refine_iter 次，上面设置为1次，进行消息传递和特征更新
        for refine_iter in range(self.mp_pair_refine_iter):
            #初始化预测的分类得分和相关性得分
            pre_cls_logits = None

            pred_relatedness_scores = None
            #启用了关系感知模块
            if self.rel_aware_on: #true
                # input_features = refine_ent_feats_each_iters[-1]

                input_features = refine_rel_feats_each_iters[-1]
                if not self.shared_pre_rel_classifier: #not false=true
                    #得到预测的关系分类得分和相关性得分
                    pre_cls_logits, pred_relatedness_scores = self.relation_conf_aware_models[
                        refine_iter
                    ](input_features, proposals, rel_pair_inds)
                else:
                    pre_cls_logits, pred_relatedness_scores = self.relation_conf_aware_models(
                        input_features, proposals, rel_pair_inds
                    )
                pre_cls_logits_each_iter.append(pre_cls_logits)
            relatedness_scores = pred_relatedness_scores

            # apply GT
            if self.apply_gt_for_rel_conf:#false
                ref_relatedness = rel_gt_binarys.clone()

                if pred_relatedness_scores is None:
                    relatedness_scores = ref_relatedness
                else:
                    relatedness_scores = pred_relatedness_scores
                    for idx, ref_rel in enumerate(ref_relatedness):
                        gt_rel_idx = ref_rel.nonzero()
                        relatedness_scores[idx][gt_rel_idx[:, 0], gt_rel_idx[:, 1]] = 1.0

            relatedness_each_iters.append(relatedness_scores)

            # build up list for massage passing process
            inst_feature4iter = [
                self.obj_downdim_fc(augment_obj_feat),
            ]
            rel_feature4iter = [
                self.rel_downdim_fc(rel_feats),
            ]

            valid_inst_idx = []
            if self.filter_the_mp_instance:#true
                for p in proposals:
                    valid_inst_idx.append(p.get_field("pred_scores") > 0.03)

            if len(valid_inst_idx) > 0:
                valid_inst_idx = torch.cat(valid_inst_idx, 0)
            else:
                valid_inst_idx = torch.zeros(0)

            self.forward_time += 1

            if self.pretrain_pre_clser_mode: #val
                #  directly return without graph building
                refined_inst_features = inst_feature4iter[-1]
                refined_rel_features = rel_feature4iter[-1]

                refine_ent_feats_each_iters.append(refined_inst_features)
                refine_rel_feats_each_iters.append(refined_rel_features)
                continue

            else:

                (
                    batchwise_rel_pair_inds,
                    subj_pred_map,
                    obj_pred_map,
                    relness_scores,
                    selected_rel_prop_pairs_idx,
                ) = self._prepare_adjacency_matrix(
                    proposals, rel_pair_inds, relatedness_each_iters[-1]
                )

                if (
                    len(squeeze_tensor(valid_inst_idx.nonzero())) < 1
                    or len(squeeze_tensor(batchwise_rel_pair_inds.nonzero())) < 1
                    or len(squeeze_tensor(subj_pred_map.nonzero())) < 1
                    or len(squeeze_tensor(obj_pred_map.nonzero())) < 1
                    or self.pretrain_pre_clser_mode
                ):  # directly return, no mp process


                    refined_inst_features = inst_feature4iter[-1]
                    refined_rel_features = rel_feature4iter[-1]

                    refine_ent_feats_each_iters.append(refined_inst_features)
                    refine_rel_feats_each_iters.append(refined_rel_features)

                    continue

            # graph module
            for t in range(self.update_step):
                param_idx = 0
                if not self.share_parameters_each_iter:
                    param_idx = t
                """update object features pass message from the predicates to instances"""
                #公式8和9
                object_sub = self.prepare_message(
                    inst_feature4iter[t],
                    rel_feature4iter[t],
                    subj_pred_map,
                    self.gate_pred2sub[param_idx],
                    relness_scores=relness_scores,
                    relness_logits=pre_cls_logits,
                )
                object_obj = self.prepare_message(
                    inst_feature4iter[t],
                    rel_feature4iter[t],
                    obj_pred_map,
                    self.gate_pred2obj[param_idx],
                    relness_scores=relness_scores,
                    relness_logits=pre_cls_logits,
                )

                GRU_input_feature_object = (object_sub + object_obj) / 2.0
                #更新实体节点，公式8，9
                inst_feature4iter.append(
                    inst_feature4iter[t]
                    + self.object_msg_fusion[param_idx](
                        GRU_input_feature_object, inst_feature4iter[t]
                    )
                )

                """update predicate features from entities features"""
                
                indices_sub = batchwise_rel_pair_inds[:, 0]
                indices_obj = batchwise_rel_pair_inds[:, 1]  # num_rel, 1

                if self.filter_the_mp_instance:#true
                    # here we only pass massage from the fg boxes to the predicates
                    valid_sub_inst_in_pairs = valid_inst_idx[indices_sub]
                    valid_obj_inst_in_pairs = valid_inst_idx[indices_obj]
                    valid_inst_pair_inds = (valid_sub_inst_in_pairs) & (
                        valid_obj_inst_in_pairs
                    )
                    # num_rel(valid sub inst), 1 Boolean tensor
                    # num_rel(valid sub inst), 1
                    indices_sub = indices_sub[valid_inst_pair_inds]
                    # num_rel(valid obj inst), 1
                    indices_obj = indices_obj[valid_inst_pair_inds]
                    ## 提取实体特征作为主语和宾语
                    feat_sub2pred = torch.index_select(inst_feature4iter[t], 0, indices_sub)
                    feat_obj2pred = torch.index_select(inst_feature4iter[t], 0, indices_obj)
                    # num_rel(valid obj inst), hidden_dim
                    valid_pairs_rel_feats = torch.index_select(
                        rel_feature4iter[t],
                        0,
                        squeeze_tensor(valid_inst_pair_inds.nonzero()),
                    )
                    ## 计算门控权重 d_s 和 d_o（公式6）
                    phrase_sub, sub2pred_gate_weight = self.gate_sub2pred[param_idx](
                        valid_pairs_rel_feats, feat_sub2pred
                    )
                    phrase_obj, obj2pred_gate_weight = self.gate_obj2pred[param_idx](
                        valid_pairs_rel_feats, feat_obj2pred
                    )
                    # 融合门控后的特征（d_s W_r^\top e_i + d_o W_r^\top e_j）
                    GRU_input_feature_phrase = (phrase_sub + phrase_obj) / 2.0
                    # 更新谓词特征（公式5）
                    next_stp_rel_feature4iter = self.pred_msg_fusion[param_idx](
                        GRU_input_feature_phrase, valid_pairs_rel_feats
                    )

                    # only update valid pairs feature, others remain as initial value
                    padded_next_stp_rel_feats = rel_feature4iter[t].clone()
                    padded_next_stp_rel_feats[
                        valid_inst_pair_inds
                    ] += next_stp_rel_feature4iter

                    rel_feature4iter.append(padded_next_stp_rel_feats)
                else:

                    # obj to pred on all pairs
                    feat_sub2pred = torch.index_select(inst_feature4iter[t], 0, indices_sub)
                    feat_obj2pred = torch.index_select(inst_feature4iter[t], 0, indices_obj)
                    phrase_sub, sub2pred_gate_weight = self.gate_sub2pred[param_idx](
                        rel_feature4iter[t], feat_sub2pred
                    )
                    phrase_obj, obj2pred_gate_weight = self.gate_obj2pred[param_idx](
                        rel_feature4iter[t], feat_obj2pred
                    )
                    GRU_input_feature_phrase = (phrase_sub + phrase_obj) / 2.0
                    rel_feature4iter.append(
                        rel_feature4iter[t]
                        + self.pred_msg_fusion[param_idx](
                            GRU_input_feature_phrase, rel_feature4iter[t]
                        )
                    )
            refined_inst_features = inst_feature4iter[-1]
            refined_rel_features = rel_feature4iter[-1]

            refine_ent_feats_each_iters.append(refined_inst_features)
            refine_rel_feats_each_iters.append(refined_rel_features)

        if (
            len(relatedness_each_iters) > 0 and not self.training
        ):  # todo why disabled in training??
            relatedness_each_iters = torch.stack(
                [torch.stack(each) for each in relatedness_each_iters]
            )
            # bsz, num_obj, num_obj, iter_num
            relatedness_each_iters = relatedness_each_iters.permute(1, 2, 3, 0)
        else:
            relatedness_each_iters = None

        if len(pre_cls_logits_each_iter) == 0:
            pre_cls_logits_each_iter = None

        return (
            refine_ent_feats_each_iters[-1],
            refine_rel_feats_each_iters[-1],
            pre_cls_logits_each_iter,
            relatedness_each_iters,
        )

class RelAwareLoss(nn.Module):
    def __init__(self):
        super(RelAwareLoss, self).__init__()
        alpha = 0.2
        gamma = 2.0

        self.pre_clser_loss_type = ('focal_fgbg_norm')

        self.predictor_type = ('hybrid')

        fgbgnorm = False
        if "fgbg_norm" in self.pre_clser_loss_type:
            fgbgnorm = True

        if "focal" in self.pre_clser_loss_type:
            self.loss_module = (
                FocalLossFGBGNormalization(alpha, gamma, fgbgnorm=fgbgnorm),
                FocalLossFGBGNormalization(alpha, gamma, fgbgnorm=fgbgnorm),
            )
        elif "bce" in self.pre_clser_loss_type:
            self.loss_module = (
                WrappedBCELoss(),
                WrappedBCELoss(),
            )

    def forward(self, pred_logit, rel_labels):
        if "focal" in self.pre_clser_loss_type:
            if self.predictor_type == "single":
                return loss_eval_mulcls_single_level(pred_logit, rel_labels, self.loss_module[0])

            elif self.predictor_type == "hybrid":
                return loss_eval_hybrid_level(pred_logit, rel_labels, self.loss_module)

        if 'bce' in self.pre_clser_loss_type:
            return  loss_eval_bincls_single_level(pred_logit, rel_labels, self.loss_module)


class BGNNPredictor(nn.Module):
    def __init__(self, args, in_channels):
        super(BGNNPredictor, self).__init__()
        self.num_obj_cls = 151
        self.num_rel_cls = 51
        self.use_bias = True

        # mode
        self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = 2048
        self.input_dim = in_channels
        self.hidden_dim = 512

        self.split_context_model4inst_rel = (False)
        self.context_layer = BGNNContext(
            args,
            self.input_dim,
            hidden_dim=self.hidden_dim,
            num_iter=3,
        )

        self.rel_feature_type = 'fusion'

        self.use_obj_recls_logits = False
        self.obj_recls_logits_update_manner = ("replace")
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        # post classification
        self.rel_classifier = build_classifier(self.hidden_dim, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.hidden_dim, self.num_obj_cls)

        self.rel_aware_model_on = True

        if self.rel_aware_model_on:
            self.rel_aware_loss_eval = RelAwareLoss()

        self.pooling_dim = 2048

        # freq
        if self.use_bias:#true
            statistics = get_dataset_statistics(args)
            self.freq_bias = FrequencyBias(None, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable

        self.init_classifier_weight()

        # for logging things
        self.forward_time = 0

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()

    def start_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode()

    def end_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode(False)

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """


        obj_feats, rel_feats, pre_cls_logits, relatedness = self.context_layer(
            roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger
        )

        if relatedness is not None:
            for idx, prop in enumerate(inst_proposals):
                prop.add_field("relness_mat", relatedness[idx])

        if self.mode == "predcls":
            obj_labels = cat(
                [proposal.get_field("labels") for proposal in inst_proposals], dim=0
            )
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat(
            [each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0
        )

        # using the object results, update the pred label and logits
        if self.use_obj_recls_logits:#false
            if self.mode == "sgdet":
                boxes_per_cls = cat(
                    [proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0
                )  # comes from post process of box_head
                # here we use the logits refinements by adding
                if self.obj_recls_logits_update_manner == "add":
                    obj_pred_logits = refined_obj_logits + obj_pred_logits
                if self.obj_recls_logits_update_manner == "replace":
                    obj_pred_logits = refined_obj_logits
                refined_obj_pred_labels = obj_prediction_nms(
                    boxes_per_cls, obj_pred_logits, nms_thresh=0.5
                )
                obj_pred_labels = refined_obj_pred_labels
            else:
                _, obj_pred_labels = refined_obj_logits[:, 1:].max(-1)
        else:
            obj_pred_labels = cat(
                [each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0
            )

        if self.use_bias:#true
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = (
                rel_cls_logits
                + self.freq_lambda * self.freq_bias.index_with_labels(pair_pred.long())
            )

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)

        add_losses = {}
        ## pre clser relpn supervision
        if pre_cls_logits is not None and self.training:
            rel_labels = cat(rel_labels, dim=0)
            for iters, each_iter_logit in enumerate(pre_cls_logits):
                if len(squeeze_tensor(torch.nonzero(rel_labels != -1))) == 0:
                    loss_rel_pre_cls = None
                else:
                    loss_rel_pre_cls = self.rel_aware_loss_eval(each_iter_logit, rel_labels)

                add_losses[f"pre_rel_classify_loss_iter-{iters}"] = loss_rel_pre_cls

        return obj_pred_logits, rel_cls_logits, add_losses


class PostProcessor_Relation(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
            self,
            attribute_on,
            use_gt_box=False,
            later_nms_pred_thres=0.3,
    ):
        """
        Arguments:

        """
        super(PostProcessor_Relation, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres

        
        self.rel_prop_on = True
        self.rel_prop_type = 'bilvl'

        self.BCE_loss = False

        self.use_relness_ranking = False
        if self.rel_prop_type == "rel_pn" and self.rel_prop_on:
            self.use_relness_ranking = False


    def forward(self, x, rel_pair_idxs, boxes):
        """
        re-NMS on refined object classifcations logits
        and ranking the relationship prediction according to the object and relationship
        classification scores

        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits = x


        rel_binarys_matrix = None
        
        if boxes[0].has_field("relness_mat"):
            rel_binarys_matrix = [ each.get_field("relness_mat") for each in boxes]

            
        if self.attribute_on:
            if isinstance(refine_logits[0], (list, tuple)):
                finetune_obj_logits, finetune_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attribute_on = False
                finetune_obj_logits = refine_logits
        else:
            finetune_obj_logits = refine_logits

        results = []
        for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
                relation_logits, finetune_obj_logits, rel_pair_idxs, boxes
        )):
            if self.attribute_on:
                att_logit = finetune_att_logits[i]
                att_prob = torch.sigmoid(att_logit)
            if not self.BCE_loss:
                obj_class_prob = F.softmax(obj_logit, -1)
            else:
                obj_class_prob = F.sigmoid(obj_logit)

            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred = obj_pred + 1
            else:
                # NOTE: by kaihua, apply late nms for object prediction
                obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit, self.later_nms_pred_thres)
                # obj_pred = box.get_field('pred_labels')
                obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                obj_scores = obj_class_prob.view(-1)[obj_score_ind]

            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            if self.use_gt_box:
                boxlist = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                device = obj_class.device
                boxes_num = obj_class.shape[0]
                regressed_box_idxs = obj_class
                boxlist = BoxList(
                    box.get_field('boxes_per_cls')[torch.arange(boxes_num, device=device), regressed_box_idxs],
                    box.size, 'xyxy')
            boxlist.add_field('pred_labels', obj_class)  # (#obj, )
            boxlist.add_field('pred_scores', obj_scores)  # (#obj, )

            if self.attribute_on:
                boxlist.add_field('pred_attributes', att_prob)

            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            rel_class_prob = F.softmax(rel_logit, -1)
            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1

            if rel_binarys_matrix is not None:
                rel_bin_mat = rel_binarys_matrix[i]
                relness = rel_bin_mat[rel_pair_idx[:, 0], rel_pair_idx[:, 1]]

            # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
            if self.use_relness_ranking:
                triple_scores = rel_scores * obj_scores0 * obj_scores1 * relness
            else:
                triple_scores = rel_scores * obj_scores0 * obj_scores1

            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            if rel_binarys_matrix is not None:
                boxlist.add_field('relness', relness[sorting_idx])
                
            boxlist.add_field('rel_pair_idxs', rel_pair_idx)  # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob)  # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels)  # (#rel, )
            results.append(boxlist)
        return results


class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, args, in_channels):
        super(ROIRelationHead, self).__init__()
        # self.cfg = args

        self.num_obj_cls = 151
        self.num_rel_cls = 51

        # mode
        self.mode = "sgdet"
        #联合视觉特征，公式2里的u
        self.union_feature_extractor = RelationFeatureExtractor(args,in_channels)
        #初始对象视觉特征提取
        self.box_feature_extractor = FPN2MLPFeatureExtractor(args, in_channels, 
                                                            half_out=False, cat_all_levels=False, for_relation=False)
        feat_dim = self.box_feature_extractor.out_channels
        #不执行下面的if
        if isinstance(self.box_feature_extractor, ResNet50Conv5ROIFeatureExtractor):
            feat_dim = self.box_feature_extractor.flatten_out_channels
    
        self.predictor = BGNNPredictor(args, feat_dim)
        self.post_processor = PostProcessor_Relation(False,False,0.5,)
        self.loss_evaluator = RelationLossComputation(False,201,10,
            True,3,False,
            [0.01858, 0.00057, 0.00051, 0.00109, 0.0015, 0.00489
            , 0.00432, 0.02913, 0.00245, 0.00121
            , 0.00404, 0.0011, 0.00132, 0.00172, 5.0e-05, 0.00242, 0.0005, 0.00048, 0.00208
            , 0.15608, 0.0265, 0.06091, 0.009, 0.00183, 0.00225
            , 0.0009, 0.00028, 0.00077, 0.04844, 0.08645, 0.31621
            , 0.00088, 0.00301, 0.00042, 0.00186, 0.001, 0.00027, 0.01012, 0.0001, 0.01286, 0.00647
            , 0.00084, 0.01077, 0.00132, 0.00069, 0.00376, 0.00214, 0.11424, 0.01205, 0.02958],
            )
        self.samp_processor = RelationSampling(0.7,False,4,1000,0.25,4096,False,False,)

        self.rel_prop_on = True
        self.rel_prop_type = (RelAwareRelFeature)

        self.object_cls_refine = False
        self.pass_obj_recls_loss = False

        # parameters
        self.use_union_box = True

        self.rel_pn_thres = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=False)
        self.rel_pn_thres_for_test = torch.nn.Parameter(
            torch.Tensor(
                [
                    0.33,
                ]
            ),
            requires_grad=False,
        )
        self.rel_pn = None
        self.use_relness_ranking = False
        self.use_same_label_with_clser = False

    def forward(self, features, proposals, targets=None, logger=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                (
                    proposals,
                    rel_labels,
                    rel_labels_all,
                    rel_pair_idxs,
                    gt_rel_binarys_matrix,
                ) = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, rel_labels_all, gt_rel_binarys_matrix = None, None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(
                features[0].device, proposals
            )

        # use box_head to extract features that will be fed to the later predictor processing
        #只提取box的视觉信息吗？
        roi_features = self.box_feature_extractor(features, proposals)


        rel_pn_loss = None
        relness_matrix = None
        if self.rel_prop_on:
            fg_pair_matrixs = None
            gt_rel_binarys_matrix = None

            if targets is not None: 
                fg_pair_matrixs, gt_rel_binarys_matrix = gt_rel_proposal_matching(
                    proposals,
                    targets,
                    0.5,
                    False,
                )
                #fg_pair_matrixs: 一个列表，其中每个元素都是一个二值矩阵，表示哪些提议框对都与真实目标框匹配。
            #gt_rel_binarys_matrix: 一个列表，其中每个元素都是一个二值矩阵，表示哪些提议框对匹配真实目标框之间的关系。
                gt_rel_binarys_matrix = [each.float().cuda() for each in gt_rel_binarys_matrix]

        if self.use_union_box:#true
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None

        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class
        rel_pn_labels = rel_labels
        if not self.use_same_label_with_clser: #true
            rel_pn_labels = rel_labels_all


        obj_refine_logits, relation_logits, add_losses = self.predictor(
            proposals,
            rel_pair_idxs,
            rel_pn_labels,
            gt_rel_binarys_matrix,
            roi_features,
            union_features,
            logger,
        )

        # proposals, rel_pair_idxs, rel_pn_labels,relness_net_input,roi_features,union_features, None
        # for test
        if not self.training:
            # re-NMS on refined object prediction logits
            if not self.object_cls_refine:
                # if don't use object classification refine, we just use the initial logits
                obj_refine_logits = [prop.get_field("predict_logits") for prop in proposals]

            result = self.post_processor(
                (relation_logits, obj_refine_logits), rel_pair_idxs, proposals
            )

            return roi_features, result, {}

        loss_relation, loss_refine = self.loss_evaluator(
            proposals, rel_labels, relation_logits, obj_refine_logits
        )

        output_losses = dict()
        if self.pass_obj_recls_loss:
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
        else:
            output_losses = dict(loss_rel=loss_relation)

        if rel_pn_loss is not None:
            output_losses["loss_relatedness"] = rel_pn_loss

        output_losses.update(add_losses)
        output_losses_checked = {}
        if self.training:
            for key in output_losses.keys():
                if output_losses[key] is not None:
                    if output_losses[key].grad_fn is not None:
                        output_losses_checked[key] = output_losses[key]
        output_losses = output_losses_checked
        return roi_features, proposals, output_losses


def build_roi_box_head(args,in_channels):
    return ROIBoxHead(args,in_channels)

def build_roi_relation_head(args,in_channels):
    return ROIRelationHead(args,in_channels)

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, args, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.args = args

    def forward(self, features, proposals, targets=None, logger=None):
        losses = {}
        x, detections, loss_box = self.box(features, proposals, targets)
        x, detections, loss_relation = self.relation(features, detections, targets, logger)
        losses.update(loss_relation)

        return x, detections, losses

def build_roi_heads(args,in_channels)->CombinedROIHeads:
    roi_heads = []
    roi_heads.append(("box", build_roi_box_head(args, in_channels)))
    roi_heads.append(("relation", build_roi_relation_head(args, in_channels)))
    roi_heads = CombinedROIHeads(args,roi_heads)
    return roi_heads