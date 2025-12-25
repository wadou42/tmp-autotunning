set -e
set -x

# Usage: ./run.sh project_name search_method init_num population evaluation_cnt repeat turn_option turn_func turn_param
# ./run.sh redis GA 10 5 50 15 true false true
# ./run.sh RANDOM RANDOM 2 1 1 1 true false true (for test)
# ./run.sh random_full_test
# ./run.sh full_test

base_dir=`pwd`
sm=$2
init_num=$3
population=$4
evaluation_cnt=$5
repeat=$6
turn_option=$7
turn_func=$8
turn_param=$9

redis_final_eva_cnt=15
doris_final_eva_cnt=15
scann_final_eva_cnt=3

redis_path="/home/scy/2025.02.13.attempt.fuction.tuning/src/redis"
scann_path="/home/scy/2025.05.23.turn.scann/src/scann/scann.O3"
scann_data_path="/home/scy/2025.05.23.turn.scann/src/scann/ann-benchmarks-base"
doris_path="/home/scy/2025.04.17.doris.exceeds.O3/src/doris.turning"

function check_sm() {
  #'GA' 'RANDOM' 'PSO' 'SA' 'DE'
  case "$sm" in
    "GA")
      ;;
    "RANDOM")
      ;;
    "PSO")
      ;;
    "SA")
      ;;
    "DE")
      ;;
    *)
      echo "Unknown argument"
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
    --pso_w 0.7 \
    --pso_c1 1.4 \
    --pso_c2 1.4 \
    --pso_v_rate 0.1 \
    --sa_temperature 100.0 \
    --sa_cooling_rate 0.95 \
    --project REDIS \
    --init_num ${init_num} \
    --default_o3_perf 1000 \
    --population ${population} \
    --evaluation_cnt ${evaluation_cnt} \
    --redis_start_core 319 \
    --redis_bench_core 311 \
    --turn_option ${turn_option} \
    --turn_func ${turn_func} \
    --turn_param ${turn_param} \
    --final_eva_cnt ${redis_final_eva_cnt} \
    --evaluate_search True \
    --evaluate_o3 False \
    --repeat ${repeat} \
    --existing_option_config_file config/start.redis.config.txt \
    --option_meta_config config/option.all.01.txt \
    --existing_ai4c_config nothing \
    --existing_constrains config/redis.all01.constrains.txt \
    --o3_project_path nothing \
    --opt_project_path ${redis_path} > log.log
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
    --turn_option ${turn_option} \
    --turn_func ${turn_func} \
    --turn_param ${turn_param} \
    --evaluate_search True \
    --evaluate_o3 False \
    --repeat ${repeat} \
    --existing_option_config_file config/start.scann.config.txt \
    --final_eva_cnt ${scann_final_eva_cnt} \
    --scann_o3_ann_dir nothing \
    --scann_opt_ann_dir ${scann_data_path} \
    --scann_o3_env nothing \
    --scann_opt_env suo.scann2 \
    --option_meta_config config/option.all.01.txt \
    --existing_ai4c_config nothing \
    --existing_constrains config/scann.all01.constrains.txt \
    --o3_project_path nothing \
    --opt_project_path ${scann_path} | tee log.log
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
  --pso_w 0.7 \
  --pso_c1 1.4 \
  --pso_c2 1.4 \
  --pso_v_rate 0.1 \
  --sa_temperature 100.0 \
  --sa_cooling_rate 0.95 \
  --project DORIS \
  --init_num ${init_num} \
  --default_o3_perf 1000 \
  --population ${population} \
  --evaluation_cnt ${evaluation_cnt} \
  --final_eva_cnt ${doris_final_eva_cnt} \
  --doris_fe_port 9030 \
  --doris_be_port 9050 \
  --turn_option ${turn_option} \
  --turn_func ${turn_func} \
  --turn_param ${turn_param} \
  --evaluate_search True \
  --evaluate_o3 False \
  --repeat ${repeat} \
  --existing_option_config_file config/start.doris.config.txt \
  --option_meta_config config/option.all.01.txt \
  --existing_ai4c_config nothing \
  --existing_constrains config/doris.all01.constrains.txt \
  --o3_project_path nothing \
  --opt_project_path ${doris_path} | tee log.log
}

function test_random_all01() {  # reuse the redis setting.
  check_sm
  cd ${base_dir}
  rm -rf random.${sm}
  cp -r src random.${sm}
  cd random.${sm}
  python -m autoturning \
    --searcher ${sm} \
    --mutation_bit_rate 0.1 \
    --crossover_bit_rate 0.1 \
    --pso_w 0.7 \
    --pso_c1 1.4 \
    --pso_c2 1.4 \
    --pso_v_rate 0.1 \
    --sa_temperature 100.0 \
    --sa_cooling_rate 0.95 \
    --project RANDOM \
    --final_eva_cnt 3 \
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
    --option_meta_config config/option.all.01.txt \
    --existing_ai4c_config nothing \
    --existing_constrains config/redis.all01.constrains.txt \
    --o3_project_path nothing \
    --opt_project_path /home/scy/2025.02.13.attempt.fuction.tuning/src/redis > log.log
}

function full_random_test() {
  for sm in 'RANDOM' 'GA' 'PSO' 'SA' 'DE'
  do
    ./run.sh RANDOM ${sm} 2 1 1 1 'true' 'false' 'true'
  done
}

function full_test() {
  for sm in 'RANDOM' 'GA' 'PSO' 'SA' 'DE'
  do
    ./run.sh RANDOM ${sm} 2 1 1 1 'true' 'false' 'true'
    ./run.sh redis ${sm} 2 1 1 1 'true' 'false' 'true'
    ./run.sh doris ${sm} 2 1 1 1 'true' 'false' 'true'
    ./run.sh scann ${sm} 2 1 1 1 'true' 'false' 'true'
  done
}

function turn_option_param() {
  for sm in 'RANDOM' 'GA' 'PSO' 'SA' 'DE'
  do
    # ./run.all.sh project_name search_method init_num population evaluation_cnt repeat turn_option turn_func turn_param
    ./run.sh redis ${sm} 10 5 50 15 'true' 'false' 'true'
    ./run.sh doris ${sm} 10 5 50 15 'true' 'false' 'true'
    ./run.sh scann ${sm} 10 5 50 15 'true' 'false' 'true'
  done
}

input=$1
case "$input" in
  "redis")
    test_redis_all01
    ;;
  "scann")
    test_scann_all01
    ;;
  "doris")
    test_doris_all01
    ;;
  "RANDOM")  # take simulate test (which is a quick test)
    test_random_all01
    ;;
  "random_full_test")  # take full test using random search
    full_random_test
    ;;
  "full_test")  # take full test using random search
    full_test
    ;;
  "turn_option_param")  # take full test using random search
    turn_option_param
    ;;
  *)
    echo "Unknown argument"
    exit 1
    ;;
esac