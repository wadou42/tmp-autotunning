#!/bin/bash
set -e
set -x

# Default values
project=""
search_method=""
init_num=10
population=5
evaluation_cnt=50
repeat=1
turn_option="true"
turn_func="false"
turn_param="false"
option_meta_config="config/optimization_filter_v5.txt"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --project)
      project="$2"
      shift 2
      ;;
    --search_method)
      search_method="$2"
      shift 2
      ;;
    --init_num)
      init_num="$2"
      shift 2
      ;;
    --population)
      population="$2"
      shift 2
      ;;
    --evaluation_cnt)
      evaluation_cnt="$2"
      shift 2
      ;;
    --repeat)
      repeat="$2"
      shift 2
      ;;
    --turn_option)
      turn_option="$2"
      shift 2
      ;;
    --turn_func)
      turn_func="$2"
      shift 2
      ;;
    --turn_param)
      turn_param="$2"
      shift 2
      ;;
    --option_meta_config)
      option_meta_config="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

base_dir=$(pwd)
sm=$search_method

function check_sm() {
  case "$sm" in
    "GA"|"RANDOM"|"PSO"|"SA"|"DE")
      ;;
    *)
      echo "Unknown search_method: $sm"
      exit 1
      ;;
  esac
}

function test_redis_all01() {
  check_sm
  cd ${base_dir}
  rm -rf redis.${sm}
  cp -r src redis.${sm}
  cd redis.${sm}
  python -m autoturning \
    --searcher ${sm} \
    --mutation_bit_rate 0.1 \
    --crossover_bit_rate 0.1 \
    --gcc_param_bit_rate 0.5 \
    --pso_w 0.7 \
    --pso_c1 1.4 \
    --pso_c2 1.4 \
    --pso_v_rate 0.1 \
    --sa_temperature 100.0 \
    --sa_cooling_rate 0.95 \
    --project REDIS \
    --turn_opt OPT \
    --init_num ${init_num} \
    --default_o3_perf 122.0595 \
    --population 5 \
    --evaluation_cnt ${evaluation_cnt} \
    --redis_start_core 319 \
    --redis_bench_core 311 \
    --turn_option ${turn_option} \
    --turn_func ${turn_func} \
    --turn_param ${turn_param} \
    --evaluate_search True \
    --final_eva_cnt 1 \
    --evaluate_o3 False \
    --repeat ${repeat} \
    --existing_option_config_file config/start.rocksdb.config.txt  \
    --option_meta_config ${option_meta_config} \
    --existing_ai4c_config nothing \
    --existing_constrains config/redis.all01.constrains.txt \
    --o3_project_path nothing \
    --opt_project_path /home/whq/dataset/redis/redis | tee log.log
}

function test_scann_all01() {
  check_sm
  cd ${base_dir}
  rm -rf scann.${sm}
  cp -r src scann.${sm}
  cd scann.${sm}
  python -m autoturning \
    --searcher ${sm} \
    --mutation_bit_rate 0.1 \
    --crossover_bit_rate 0.1 \
    --gcc_param_bit_rate 0.5 \
    --pso_w 0.7 \
    --pso_c1 1.4 \
    --pso_c2 1.4 \
    --pso_v_rate 0.1 \
    --sa_temperature 100.0 \
    --sa_cooling_rate 0.95 \
    --project SCANN \
    --init_num ${init_num} \
    --default_o3_perf "457.58623674802215" \
    --population ${population} \
    --evaluation_cnt ${evaluation_cnt} \
    --final_eva_cnt 3 \
    --turn_option ${turn_option} \
    --turn_func ${turn_func} \
    --turn_param ${turn_param} \
    --evaluate_search True \
    --evaluate_o3 False \
    --repeat ${repeat} \
    --existing_option_config_file config/prediction_result/prediction_scann.txt \
    --scann_o3_ann_dir nothing \
    --scann_opt_ann_dir /home/whq/scann/scann/ann-benchmarks-base \
    --scann_o3_env nothing \
    --scann_opt_env scann \
    --option_meta_config ${option_meta_config} \
    --existing_ai4c_config nothing \
    --existing_constrains config/scann.all01.constrains.txt \
    --o3_project_path nothing \
    --opt_project_path /home/whq/scann/scann/scann | tee log.log
}

function test_doris_all01() {
  check_sm
  cd ${base_dir}
  rm -rf doris.${sm}
  cp -r src doris.${sm}
  cd doris.${sm}
  python -m autoturning \
  --searcher ${sm} \
  --mutation_bit_rate 0.1 \
  --crossover_bit_rate 0.1 \
  --gcc_param_bit_rate 0.5 \
  --pso_w 0.7 \
  --pso_c1 1.4 \
  --pso_c2 1.4 \
  --pso_v_rate 0.1 \
  --sa_temperature 100.0 \
  --sa_cooling_rate 0.95 \
  --project DORIS \
  --init_num ${init_num} \
  --default_o3_perf 2417 \
  --population ${population} \
  --final_eva_cnt 1 \
  --evaluation_cnt ${evaluation_cnt} \
  --doris_fe_port 9030 \
  --doris_be_port 9050 \
    --turn_option ${turn_option} \
    --turn_func ${turn_func} \
    --turn_param ${turn_param} \
  --evaluate_search True \
  --evaluate_o3 False \
  --repeat ${repeat} \
  --existing_option_config_file config/ir2vec.start.doris.config.txt \
  --option_meta_config ${option_meta_config} \
  --existing_ai4c_config nothing \
  --existing_constrains config/doris.all01.constrains.txt \
  --o3_project_path nothing \
  --opt_project_path /home/whq/dataset/dorisO3 | tee log.log
}

function test_rocksdb_all01() {
  check_sm
  cd ${base_dir}
  rm -rf rocksdb.${sm}
  cp -r src rocksdb.${sm}
  cd rocksdb.${sm}
  python -m autoturning \
  --searcher ${sm} \
  --mutation_bit_rate 0.1 \
  --crossover_bit_rate 0.1 \
  --gcc_param_bit_rate 0.5 \
  --pso_w 0.7 \
  --pso_c1 1.4 \
  --pso_c2 1.4 \
  --pso_v_rate 0.1 \
  --sa_temperature 100.0 \
  --sa_cooling_rate 0.95 \
  --project ROCKSDB \
  --init_num ${init_num} \
  --default_o3_perf 114054.05052215316 \
  --population ${population} \
  --final_eva_cnt 1 \
  --evaluation_cnt ${evaluation_cnt} \
    --turn_option ${turn_option} \
    --turn_func ${turn_func} \
    --turn_param ${turn_param} \
  --evaluate_search True \
  --evaluate_o3 False \
  --repeat ${repeat} \
  --existing_option_config_file config/start.rocksdb.config.txt \
  --option_meta_config ${option_meta_config} \
  --existing_ai4c_config nothing \
  --o3_project_path nothing \
  --opt_project_path /home/whq/dataset/rocksdb.6.26.1 | tee log.log
  # --opt_project_path /home/whq | tee log.log
}

function test_zstd_all01() {
  check_sm
  cd ${base_dir}
  rm -rf zstd.${sm}
  cp -r src zstd.${sm}
  cd zstd.${sm}
  python -m autoturning \
  --searcher ${sm} \
  --mutation_bit_rate 0.1 \
  --crossover_bit_rate 0.1 \
  --gcc_param_bit_rate 0.5 \
  --pso_w 0.7 \
  --pso_c1 1.4 \
  --pso_c2 1.4 \
  --pso_v_rate 0.1 \
  --sa_temperature 100.0 \
  --sa_cooling_rate 0.95 \
  --project ZSTD \
  --init_num ${init_num} \
  --default_o3_perf 232.0 \
  --population ${population} \
  --final_eva_cnt 10 \
  --evaluation_cnt ${evaluation_cnt} \
    --turn_option ${turn_option} \
    --turn_func ${turn_func} \
    --turn_param ${turn_param} \
  --evaluate_search True \
  --evaluate_o3 False \
  --repeat ${repeat} \
  --existing_option_config_file config/start.rocksdb.config.txt \
  --option_meta_config ${option_meta_config} \
  --existing_ai4c_config nothing \
  --o3_project_path nothing \
  --opt_project_path /home/whq/workspace/zstd-instance5 | tee log.log
  # --opt_project_path /home/whq | tee log.log
}

function test_mysql_all01() {
  check_sm
  cd ${base_dir}
  rm -rf mysql.${sm}
  cp -r src mysql.${sm}
  cd mysql.${sm}
  python -m autoturning \
  --searcher ${sm} \
  --mutation_bit_rate 0.1 \
  --crossover_bit_rate 0.1 \
  --gcc_param_bit_rate 0.5 \
  --pso_w 0.7 \
  --pso_c1 1.4 \
  --pso_c2 1.4 \
  --pso_v_rate 0.1 \
  --sa_temperature 100.0 \
  --sa_cooling_rate 0.95 \
  --project MYSQL \
  --init_num ${init_num} \
  --default_o3_perf 232.0 \
  --population ${population} \
  --final_eva_cnt 10 \
  --evaluation_cnt ${evaluation_cnt} \
    --turn_option ${turn_option} \
    --turn_func ${turn_func} \
    --turn_param ${turn_param} \
  --evaluate_search True \
  --evaluate_o3 False \
  --repeat ${repeat} \
  --mysql_install_home /home/whq/bin/mysql \
  --mysql_data_home /home/whq/data/mysql \
  --mysql_test_home /home/whq/dataset/mysql/sysbench \
  --mysql_cnf_file /etc/my.cnf \
  --existing_option_config_file config/start.mysql.config.txt \
  --existing_constrains config/mysql.all01.constrains.txt \
  --option_meta_config ${option_meta_config} \
  --existing_ai4c_config nothing \
  --o3_project_path nothing \
  --opt_project_path /home/whq/bin/mysql | tee log.log
}

function test_random_all01() {  # reuse the redis setting.
  check_sm
  cd ${base_dir}
  rm -rf random.${sm}
  cp -r src random.${sm}
  cd random.${sm}
  python -m autoturning \
    --searcher ${sm} \
    --mutation_bit_rate 0.3 \
    --crossover_bit_rate 0.1 \
    --gcc_param_bit_rate 0.5 \
    --pso_w 0.7 \
    --pso_c1 1.4 \
    --pso_c2 1.4 \
    --pso_v_rate 0.1 \
    --sa_temperature 100.0 \
    --sa_cooling_rate 0.95 \
    --project RANDOM \
    --init_num ${init_num} \
    --default_o3_perf 1000 \
    --population ${population} \
    --evaluation_cnt ${evaluation_cnt} \
    --redis_start_core 319 \
    --redis_bench_core 311 \
    --turn_option ${turn_option} \
    --turn_func ${turn_func} \
    --turn_param ${turn_param} \
    --evaluate_search True \
    --evaluate_o3 False \
    --repeat ${repeat} \
    --existing_option_config_file config/start.redis.config.txt \
    --option_meta_config ${option_meta_config} \
    --existing_ai4c_config nothing \
    --existing_constrains config/redis.all01.constrains.txt \
    --o3_project_path nothing \
    --opt_project_path /home/whq/dataset/redis/redis > log.log
}

function full_random_test() {
  for sm in 'RANDOM' 'GA' 'PSO' 'SA' 'DE'
  do
    ./run.all.sh --project rocksdb --search_method ${sm} --init_num 10 --population 5 --evaluation_cnt 50 --repeat 3 --turn_option true --turn_func false --turn_param false --option_meta_config config/option.txt
  done
}

function full_test() {
  for sm in 'RANDOM' 'GA' 'PSO' 'SA' 'DE'
  do
    ./run.all.sh --project RANDOM --search_method ${sm} --init_num 2 --population 1 --evaluation_cnt 1 --repeat 1 --turn_option true --turn_func false --turn_param true
    ./run.all.sh --project redis --search_method ${sm} --init_num 2 --population 1 --evaluation_cnt 1 --repeat 1 --turn_option true --turn_func false --turn_param true
    ./run.all.sh --project doris --search_method ${sm} --init_num 2 --population 1 --evaluation_cnt 1 --repeat 1 --turn_option true --turn_func false --turn_param true
    ./run.all.sh --project scann --search_method ${sm} --init_num 2 --population 1 --evaluation_cnt 1 --repeat 1 --turn_option true --turn_func false --turn_param true
  done
}

function turn_option_param() {
  for sm in 'RANDOM' 'GA' 'PSO' 'SA' 'DE'
  do
    ./run.all.sh --project redis --search_method ${sm} --init_num 10 --population 5 --evaluation_cnt 50 --repeat 15 --turn_option true --turn_func false --turn_param true
    ./run.all.sh --project doris --search_method ${sm} --init_num 10 --population 5 --evaluation_cnt 50 --repeat 15 --turn_option true --turn_func false --turn_param true
    ./run.all.sh --project scann --search_method ${sm} --init_num 10 --population 5 --evaluation_cnt 50 --repeat 15 --turn_option true --turn_func false --turn_param true
  done
}

input=$1
case "$project" in
  "redis")
    test_redis_all01
    ;;
  "scann")
    test_scann_all01
    ;;
  "doris")
    test_doris_all01
    ;;
  "rocksdb")
    test_rocksdb_all01
    ;;
  "zstd")
    test_zstd_all01
    ;;
  "mysql")
    test_mysql_all01
    ;;
  "RANDOM")
    test_random_all01
    ;;
  "random_full_test")
    full_random_test
    ;;
  "full_test")
    full_test
    ;;
  "turn_option_param")
    turn_option_param
    ;;
  *)
    echo "Unknown or missing --project argument"
    exit 1
    ;;
esac

# 使用样例：
# bash run.all.sh --project redis --search_method SA --init_num 10 --population 5 --evaluation_cnt 50 --repeat 15 --turn_option true --turn_func false --turn_param true --option_meta_config config/optimization_filter_v5.txt
# bash run.all.sh --project doris --search_method SA --init_num 10 --population 5 --evaluation_cnt 50 --repeat 15 --turn_option true --turn_func false --turn_param true --option_meta_config config/optimization_filter_v5.txt
# bash run.all.sh --project scann --search_method SA --init_num 10 --population 5 --evaluation_cnt 50 --repeat 3 --turn_option true --turn_func false --turn_param true --option_meta_config config/optimization_filter_v5.txt
# bash run.all.sh --project rocksdb --search_method SA --init_num 10 --population 5 --evaluation_cnt 50 --repeat 3 --turn_option true --turn_func false --turn_param false --option_meta_config config/option.txt
# bash run.all.sh --project zstd --search_method SA --init_num 10 --population 5 --evaluation_cnt 50 --repeat 3 --turn_option true --turn_func false --turn_param false --option_meta_config config/option.txt
# bash run.all.sh --project mysql --search_method SA --init_num 10 --population 5 --evaluation_cnt 50 --repeat 3 --turn_option true --turn_func false --turn_param false --option_meta_config config/option.txt
