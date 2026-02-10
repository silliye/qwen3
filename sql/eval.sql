SET odps.instance.priority = 1;

-- DROP TABLE IF EXISTS sp_ann_llm_cy_pretrain_eval_table_aug_v1;

CREATE TABLE IF NOT EXISTS sp_ann_llm_cy_pretrain_eval_table_aug_v1
(
    sample_id               STRING
   ,target_item_id          STRING
   ,predict_item_token      STRING
   ,cur_relax_num           STRING  -- 注意：实际逻辑中这里似乎写入了计算后的Score
   ,levels                  STRING
)
PARTITIONED BY
(
    ds                      STRING
   ,nation                  STRING
   ,cdvs                    STRING
   ,topn                    STRING
   ,candi_num               STRING
   ,relax_num               STRING
)
LIFECYCLE 60
;

INSERT OVERWRITE TABLE sp_ann_llm_cy_pretrain_eval_table_aug_v1 PARTITION (ds = "${ds}",nation = '${nation}',cdvs = '${cdvs}', topn='${topn}', candi_num='${candi_num}', relax_num='${relax_num}')
SELECT sample_id
      ,SPLIT(sample_id,"&&")[1] AS target_item_id
      ,predict_classes AS predict_item_token
      -- half: parallel: 打分机制 从全乘法到 乘法&加法
      -- 下面这行是被注释掉的旧逻辑:
      -- ,8 - REGEXP_COUNT(predict_classes, '\\*') + DoubleArrayMulADD(TRANSFORM(split(predict_probs, ","), x -> if(x='*', '1.0', x)))
      -- 其他 (当前生效的逻辑):
      ,8 - REGEXP_COUNT(predict_classes, '\\*') + DoubleArrayMul(TRANSFORM(split(predict_probs, ","), x -> if(x='*', '1.0', x))) AS cur_relax_num
      ,"${levels}" AS levels
FROM
    (
        SELECT sample_id
              ,combo.`0` AS predict_classes
              ,combo.`1` AS predict_probs
        FROM
            (
                SELECT sample_id
                      ,GenerateREGCombinations(SPLIT(combo.`0`,","), ${relax_num}) AS predict_classes_list_aug
                      ,GenerateREGCombinations(SPLIT(combo.`1`,","), ${relax_num}) AS predict_probs_list_aug
                FROM
                    (
                        SELECT sample_id
                              -- 这里的 CheckREGArraySplitV1 似乎是用来做截断和分割的
                              -- 参数: sequence, levels, topn
                              -- SLICE(..., 1, candi_num) 取前 candi_num 个
                              ,SLICE(CheckREGArraySplitV1(user_seq_predict_2step_top50,"${levels}","${topn}"),1,"${candi_num}") AS predict_classes_list
                              -- 注意: 这里用了 CheckREGArraySplitV2 处理 token_level1_target (推测这里实际存的是probs)
                              ,SLICE(CheckREGArraySplitV2(token_level1_target,"${levels}","${topn}"),1,"${candi_num}") AS predict_probs_list
                        FROM   i18n_algo_dev.cyllm_sp_ann_xdl2_test_v1_eval_trace_mul_nd_v2_pretrain
                        WHERE  ds = "${ds}"
                        AND    nation = '${nation}'
                        AND    cdvs = '${cdvs}'
                        AND    RAND() < 0.20  -- 采样 20% 数据进行 Eval
                    )
                LATERAL VIEW POSEXPLODE(ARRAYS_ZIP(predict_classes_list, predict_probs_list)) posExploded AS pos, combo
            )
        LATERAL VIEW POSEXPLODE(ARRAYS_ZIP(predict_classes_list_aug, predict_probs_list_aug)) y AS pos, combo
    )
GROUP BY sample_id, predict_classes, predict_probs
;

