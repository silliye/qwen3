CREATE TABLE IF NOT EXISTS sp_ann_llm_cy_pretrain_infer_table_aug_v1
(
    sample_id               STRING
   ,real_target_token_aug   STRING
)
PARTITIONED BY
(
    ds                      STRING
   ,nation                  STRING
   ,cdvs                    STRING
   ,relax_num               STRING
)
LIFECYCLE 60
;

INSERT OVERWRITE TABLE sp_ann_llm_cy_pretrain_infer_table_aug_v1 PARTITION (ds = "${ds}",nation = '${nation}',cdvs = '${cdvs}', relax_num = '${relax_num}')
SELECT sample_id
      ,target_token AS real_target_token_aug
FROM
    (
        SELECT sample_id
        -- 返回很多结果，事实上一个sample_id对应多个结果
              ,GenerateREGCombinations(SPLIT(target_token,","),${relax_num}) AS target_token_list_aug    
                      FROM
            (
                SELECT sample_id
                      ,CONCAT_WS(",",TRANSFORM(SPLIT(token_level3_target,","),x -> CAST(CAST(x AS BIGINT) AS STRING))) AS target_token
                FROM   i18n_algo_dev.cyllm_sp_ann_xdl2_full_infer_trace_mul_nd_v2_pretrain
                WHERE  ds = "${ds}"
                AND    nation = '${nation}'
                AND    cdvs = '${cdvs}'
            )
    )
    -- 行转列，GenerateREGCombinations 返回的是一个list，可以理解为，然后转成列和对应的一个sample id一起聚合
LATERAL VIEW EXPLODE(target_token_list_aug) y AS target_token
;

-- infer是负责把 松弛后的结果写到一个表中; 作为真实表去用;
-- 比如4kw个sample_id 松弛后可能有96 * 4kw个可能