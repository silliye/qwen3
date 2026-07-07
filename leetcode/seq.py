import os
import sys
import json
import yaml
import argparse
import base64
import collections
import uuid
import socket
import functools
import itertools
from datetime import datetime
import heapq

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '../../../../../dragon'))

from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.rfm.rfm_api_mixin import RfmApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin
from dragonfly.ext.ad_union_train.ad_union_train_api_mixin import AdUnionTrainApiMixin
from dragonfly.decorators import parallel
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin
from dragonfly.matx.dragonfly_context import DragonflyContext
from typing import Dict, Callable, Tuple, Type
from typing import ByteString as bytes_view
from typing import List as FTList
from typing import Dict as FTDict
from typing import Set as FTSet
import json
import math
import random

kuiba_list_converter_config_limit50 = {"converter": "list", "converter_args": {"reversed": False, "enable_filter": False, "limit": 50}}
kuiba_list_converter_config_limit100 = {"converter": "list", "converter_args": {"reversed": False, "enable_filter": False, "limit": 100}}
kuiba_list_converter_config_limit2048 = {"converter": "list", "converter_args": {"reversed": False, "enable_filter": False, "limit": 2048}}

class GSUServerFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, CofeaApiMixin, EmbedCalcApiMixin, KgnnApiMixin, UniPredictApiMixin, RfmApiMixin, AdUnionTrainApiMixin, EmbeddingApiMixin):
  def gpt_gsu(self):
    self \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "time_ms",
          "to_common_attr": "id_list",
          "aggregator" : "min",
          "to_common_attr": "min_time_ms",
          "pack_if": "time_ms"
        }]
      ) \
      .enrich_attr_by_lua(
        import_common_attr=["_REQ_TIME_", "min_time_ms"],
        export_common_attr=["min_time_ms_zero", "request_gt_min", "request_gt600_min","request_diff"],
        function_for_common="calculate",
        lua_script="""
          function calculate()
            if min_time_ms == nil then
              return 1, 0, 0, 0
            end
            local diff = (_REQ_TIME_ - min_time_ms)/60000
            if diff >= 10 then
              return 0, 1, 1, diff
            end
            if diff >= 0 then
              return 0, 1, 0, diff
            end
            return 1, 0, 0, 0
          end
        """
      ) \
      .perflog_attr_value(check_point="recogpt.time_stat_diff", common_attrs=["min_time_ms_zero", "request_gt_min", "request_gt600_min"]) \
      .copy_user_meta_info(save_request_time_to_attr="origin_request_time") \
      .if_("request_gt_min > 0") \
        .debug_log(common_attrs=["_REQ_TIME_", "origin_request_time",  "min_time_ms", "request_gt_min","request_diff"], log_tag="before_truncate", respect_sample_logging=False) \
        .perflog_attr_value(check_point="recogpt.time_stat_diff", common_attrs=["request_diff"]) \
      .end_if_() \
      .reset_user_meta_info(timestamp_attr="min_time_ms", time_unit="ms")

    self \
      .gsu_common_colossusv2_enricher(
          kconf="colossus.kconf_client.video_item",
          item_fields={
              "photo_id" : "colossus_photo_id",
              "author_id_v2" : "colossus_author_id_v2",
              "timestamp" : "colossus_timestamp",
              "label" : "colossus_label",
              "duration" : "colossus_duration",
              "play_time" : "colossus_play_time",
              "channel" : "colossus_channel",
              "tag" : "colossus_tag",
          },
          filter_future_items=True,
          seconds_to_lookback=600,
          limit=2048,
      ) \
      .fetch_remote_embedding(
        protocol=1,
        colossusdb_embd_model_name="lyj_fr_tw_semantic_id",
        colossusdb_embd_table_name="emb_lyj_fr_tw_pid2sid",
        id_converter={"type_name": "plainIdConverter"},
        input_attr_name="colossus_photo_id",
        output_attr_name="colossus_photo_id_sids",
        query_source_type="common_attr",
        is_raw_data=True,
        raw_data_type="uint16",
        timeout_ms=50,
        size=3,
        max_signs_per_request=500,
      ) \
      .enrich_attr_by_lua(
        import_common_attr=["colossus_photo_id_sids"],
        export_common_attr=["colossus_photo_id_sids_fix0", "colossus_photo_id_sids_fix1", "colossus_photo_id_sids_fix2", "colossus_photo_id_sids_hit"],
        function_for_common="calc",
        lua_script="""
            function calc()
                local colossus_photo_id_sids_fix0 = {}
                local colossus_photo_id_sids_fix1 = {}
                local colossus_photo_id_sids_fix2 = {}
                if colossus_photo_id_sids == nil then
                  return colossus_photo_id_sids_fix0, colossus_photo_id_sids_fix1, colossus_photo_id_sids_fix2, 1.0
                end
                local seq_len = (#colossus_photo_id_sids // 3) - 1
                local seq_hit_len = 0.0
                for i = 0, seq_len do
                  if colossus_photo_id_sids[3*i+1] == 0 and colossus_photo_id_sids[3*i+2] == 0 and colossus_photo_id_sids[3*i+3] == 0 then
                    colossus_photo_id_sids_fix0[i+1] = 100000
                    colossus_photo_id_sids_fix1[i+1] = 100000
                    colossus_photo_id_sids_fix2[i+1] = 100000
                  else
                    colossus_photo_id_sids_fix0[i+1] = colossus_photo_id_sids[3*i+1]
                    colossus_photo_id_sids_fix1[i+1] = colossus_photo_id_sids[3*i+2]
                    colossus_photo_id_sids_fix2[i+1] = colossus_photo_id_sids[3*i+3]
                    seq_hit_len = seq_hit_len + 1.0
                  end
                end
                return colossus_photo_id_sids_fix0, colossus_photo_id_sids_fix1, colossus_photo_id_sids_fix2, seq_hit_len / (seq_len+1)
            end
        """
      ) \
      .perflog_attr_value(check_point="recorandom.colossus_check", common_attrs=["colossus_photo_id_sids_hit"]) \

    
    self \
      .copy_item_meta_info(save_item_seq_to_attr="item_seq") \
      .enrich_attr_by_lua(
        import_common_attr=["colossus_play_time", "colossus_duration", "_REQ_TIME_", "colossus_timestamp"],
        export_common_attr=["colossus_play_x_duration", "colossus_day_diff", "colossus_hour_diff", "colossus_len"],
        function_for_common ="calculate",
        lua_script="""
            function calculate()
                local play_x_duration = {}
                local day_diff = {}
                local hour_diff = {}
                local colossus_len = 0.0

                if colossus_play_time ~= nil and colossus_duration ~= nil and colossus_timestamp ~= nil then
                    colossus_len = #colossus_play_time
                    for i = 1, #colossus_play_time do
                        index = #colossus_play_time - i + 1
                        local play_time = math.min(colossus_play_time[index], (1 << 24) - 1)
                        local duration = math.min(colossus_duration[index], (1 << 24) - 1)
                        play_x_duration[index] = (play_time << 24) + duration
                        day_diff[index] = (_REQ_TIME_ // 1000 - colossus_timestamp[index]) // (24 * 3600)
                        hour_diff[index] = ((_REQ_TIME_ // 1000 - colossus_timestamp[index]) // 3600) % 24
                    end
                end
                return play_x_duration, day_diff, hour_diff, colossus_len
            end
        """) \
      .perflog_attr_value(check_point="recogpt.colossus_return", common_attrs=["colossus_len",]) \
      .log_debug_info(for_debug_request_only=False,common_attrs = ["colossus_day_diff"]) \
      .log_debug_info(for_debug_request_only=False,common_attrs = ["colossus_timestamp"]) \


    self \
      .extract_kuiba_parameter(
        config={
          "extract_colossus_field_photo_id": {"attrs": [{"key_type": 26, "mio_slot_key_type": 1040, "attr": ["colossus_photo_id"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_author_id_v2": {"attrs": [{"key_type": 128, "mio_slot_key_type": 1041, "attr": ["colossus_author_id_v2"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_tag": {"attrs": [{"key_type": 349, "mio_slot_key_type": 1042, "attr": ["colossus_tag"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_sids0": {"attrs": [{"key_type": 410, "mio_slot_key_type": 1043, "attr": ["colossus_photo_id_sids_fix0"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_sids1": {"attrs": [{"key_type": 411, "mio_slot_key_type": 1044, "attr": ["colossus_photo_id_sids_fix1"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_sids2": {"attrs": [{"key_type": 412, "mio_slot_key_type": 1045, "attr": ["colossus_photo_id_sids_fix2"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_duration": {"attrs": [{"key_type": 402, "mio_slot_key_type": 1046, "attr": ["colossus_duration"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_play_time": {"attrs": [{"key_type": 401, "mio_slot_key_type": 1047, "attr": ["colossus_play_time"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_play_x_duration": {"attrs": [{"key_type": 348, "mio_slot_key_type": 1048, "attr": ["colossus_play_x_duration"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_label": {"attrs": [{"key_type": 696, "mio_slot_key_type": 1049, "attr": ["colossus_label"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_channel": {"attrs": [{"key_type": 700, "mio_slot_key_type": 1050, "attr": ["colossus_channel"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_day_diff": {"attrs": [{"key_type": 350, "mio_slot_key_type": 1051, "attr": ["colossus_day_diff"], **kuiba_list_converter_config_limit2048,}],},
          "extract_colossus_field_hour_diff": {"attrs": [{"key_type": 1909, "mio_slot_key_type": 1052, "attr": ["colossus_hour_diff"], **kuiba_list_converter_config_limit2048,}],},
        },
        target_item={"item_seq": 0},
        is_common_attr=False,
        slot_as_attr_name=True)
    
    self \
      .copy_attr(
        attrs=[{
          "from_common": "colossus_timestamp",
          "to_item": "colossus_time_s"
        }],
        target_item={"item_seq": 0},
      )

    return self

predict_for_gsu = GSUServerFlow(name = "predict_for_gsu").gpt_gsu()

service = LeafService(kess_name="grpc_ypq_recent_2k_seq_server",
                      item_attrs_from_request=["photo_id", "16346", "13346", "time_ms",],
                      common_attrs_from_request=["246",],
                    )

service.return_item_attrs([
  "1040", "1041", "1042", \
  "1043", "1044", "1045", \
  "1046", "1047", "1048", \
  "1049", "1050", "1051", \
  "1052", "colossus_time_s", \
])

service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False

service.PY_ENABLE_REMOTE_COMPILE = True
service.PY_USE_REMOTE_DSO = True 

service.add_leaf_flows(leaf_flows = [predict_for_gsu], request_type = "predict_for_gsu")

if __name__ == '__main__':
  out_file = str(__file__).replace('py', 'json')
  service.build(output_file=os.path.join(current_dir, out_file))