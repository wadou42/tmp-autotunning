# 项目简介


## 功能介绍
项目的功能主要是为Redis、Doris、ScaNN调优。

项目的输入为config文件夹下的配置文件，主要包括：
1. config/option.all.01.txt：参与调优的优化选项文件，使用--option_meta_config选项指定。
2. config/${project}.all01.constrains.txt：编译约束文件，使用--existing_constrains选项指定。
3. config/start.${project}.config.txt：搜索起始位置，使用--existing_option_config_file选项指定。

项目的输出为调优日志，主要包括：
1. opt.recorder.txt：优化记录，格式为”迭代次数: 优化“。
2. perf.recorder.txt：性能记录，格式为”迭代次数：O3_perf=xxx, opt_perf=xxx, acc_rate=xxx“
3. final.perf.recorder.txt：最佳优化复现记录。

## 项目原理

下面简述了项目支持的调优配置参与调优的过程：
1. 调优配置（优化选项字符串）通过parser转换为metadata（用于记录配置项的基本信息）和config（dict，用于记录一个调优配置，如一个优化选项字符串）。
2. config通过encoder转换为vector，vector直接作为searcher的输入进行调优。
3. 在searcher调优过程中，需要通过 vector -> decoder -> config -> writer -> 优化配置，完成从向量向调优配置的转换。
4. evaluator接受调优配置，并以此编译待调优项目进行性能评估。
5. evaluator将结果反馈给searcher，searcher进行进一步搜索，直到达到搜索条件限制。

# 主要文件

```commandline
src
│  README.md        README文件
│
├─autoturning       项目代码目录
│  │  codec.py        encoder-decoder，负责将config转化为直接参与调优的向量
│  │  config_parser.py               负责生成config，随机生成config，或者从文件中读取config
│  │  config_writer.py               负责将config转换为编译选项
│  │  constrains_solver.py           编译约束处理器
│  │  evaluator.py                   负责性能评估
│  │  metadata.py                    元信息定义，负责存储字段名、字段类型、上下界等
│  │  metadata_parser.py             负责从文件中读取元信息（控制参与调优的字段）
│  │  recorder.py                    负责记录调优信息
│  │  script_args.py                 负责记录全局参数
│  │  searcher.py                    负责搜索
│  │  utils.py                       工具文件，其中封装了计算加速比，向量规范化（因为搜索可能会超出限制），向量变异等。
│  │  __main__.py                    主文件。
│  │
│  └─manager        封装对doris、redis、scann的编译+测试
│          doris_manager.py          封装对doris的编译+测试
│          manager.py                manager的通用定义
│          redis_manager.py          封装对redis的编译+测试
│          scann_manager.py          封装对scann的编译+测试
│
└─config            配置文件
        doris.all01.constrains.txt    doris的01选项约束
        option.all.01.txt             所有01选项
        redis.all01.constrains.txt    redis的01选项约束
        scann.all01.constrains.txt    scann的01选项约束
        start.redis.config.txt        redis起始点
        start.scann.config.txt        doris起始点
```
注：没有在上述记录中出现的文件用于调试。

# 环境依赖

```commandline
pymysql                   1.1.1                    pypi_0    pypi
psutil                    7.0.0                    pypi_0    pypi
redis                     5.2.1                    pypi_0    pypi
```

# 使用方法

```commandline
# 将run.all.sh和src放在一个目录下。
./run.all.sh [redis, doris, scann] [GA, RANDOM, PSO, SA, DE]
```

# 参数介绍

```commandline
--option_meta_config OPTION_META_CONFIG
                    搜索启用的编译选项. (default: config/option.txt)
--existing_constrains EXISTING_CONSTRAINS
                    编译约束文件。 (default: None)
--existing_option_config_file EXISTING_OPTION_CONFIG_FILE
                    优化搜索起点文件。 (default: None)
--searcher SEARCHER   搜索器: ['GA', 'RANDOM', 'PSO', 'SA', 'DE'] (default: None)
--evaluation_cnt_limit EVALUATION_CNT_LIMIT
                    迭代次数限制. (default: 50)
--time_limit TIME_LIMIT
                    时间限制. (default: 82800)
--mutation_bit_rate MUTATION_BIT_RATE
                    变异率，作用于具有变异行为的搜索器：['GA', 'RANDOM', 'SA', 'DE']. (default: None)
--crossover_bit_rate CROSSOVER_BIT_RATE
                    交叉互换率，作用于具有交叉互换行为的搜索器：['GA', 'DE'] (default: None)
--init_num INIT_NUM   初始起点个数。 (default: 10)
--population POPULATION
                    种群数量/粒子数量. (default: 5)
--default_o3_perf DEFAULT_O3_PERF
                    默认O3性能. (default: None)
--project PROJECT     调优项目： ['DORIS', 'REDIS', 'SCANN', 'RANDOM', 'FAST_DORIS'] (default: None)
--o3_project_path O3_PROJECT_PATH
                    利用O3进行编译的项目路径. (default: None)
--opt_project_path OPT_PROJECT_PATH
                    利用优化选项进行编译的项目路径. (default: None)
--scann_o3_ann_dir SCANN_O3_ANN_DIR
                    利用O3进行编译的ScaNN使用的benchmark的目录 (default: None)
--scann_opt_ann_dir SCANN_OPT_ANN_DIR
                    利用优化选项进行编译的ScaNN使用的benchmark的目录 (default: None)
--scann_o3_env SCANN_O3_ENV
                    利用O3进行编译的ScaNN安装的Conda环境 (default: None)
--scann_opt_env SCANN_OPT_ENV
                    利用优化选项进行编译的ScaNN安装的Conda环境 (default: None)
--final_eva_cnt FINAL_EVA_CNT
                    最终评估中的重复次数（项目的实际执行次数会是manager中指定的次数 * 这个次数）。 (default: 50)
--opt_recorder OPT_RECORDER
                    优化选项记录文件，会记录用于编译的选项. (default: opt.recorder.txt)
--perf_recorder PERF_RECORDER
                    性能记录文件. (default: perf.recorder.txt)
--final_perf_recorder FINAL_PERF_RECORDER
                    最终评估的性能记录文件. (default: final.perf.recorder.txt)
--clean CLEAN         是否清除已有结果. (default: True)
--turn_what TURN_WHAT
                    调优目标，目前只能选OPT： ['OPT', 'FUNC', 'OPT_FUNC'] (default: None)
--turn_opt TURN_OPT   是否编译并测试O3，若选择O3_OPT则代表编译并测试O3（实际上并无必要，因为指定了default_o3_perf） ['OPT', 'O3_OPT'] (default: None)
```

# TDOOs

实际上项目在Doris上尝试过使用剪枝后的AI4C配置文件（因为使用完整的AI4C配置文件会爆内存）进行调优，且功能正常。
但Redis/ScaNN上并没有相关的适配和测试。

项目设计之初考虑了枚举类型以及值类型的编译选项调优，但该类选项面临的主要问题在于：约束难找。
对于枚举类型的选项，也许直接将其视为一个完整的选项，对构建失败的选项进行约简即可。
然而对于值类型的选项，也许需要先进行约简，再对其取值进行二分查找。
但无论如何，寻找项目编译约束都是耗时巨大的事情。