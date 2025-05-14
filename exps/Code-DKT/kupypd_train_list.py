
import os
import time

from kupypd_cache_data import CacheDataLoad
from kupypd_train_eval import train


import sys
module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)
import src.make_logger as make_logger

def train_list(gpu_devices, is_hp_search, train_info_list, folds, logger, logger_name, logger_dir, debug):
    if debug or len(gpu_devices) < 2:
        if len(gpu_devices)>=1:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices[0]
        cached_data = CacheDataLoad(logger, debug)
        for train_info in train_info_list:
            try:
                train(**train_info, cached_data=cached_data, folds=folds, is_hp_search=is_hp_search, logger=logger, debug=debug)
            except Exception as e:
                make_logger.write_exception_log(logger, e)
                exit(-1)

            logger.info("")
    else:
        logger.info("multiprocessing.start")

        import multiprocessing

        shared_lock = multiprocessing.Lock()

        manager = multiprocessing.Manager()
        shared_check_dict = manager.dict()

        def train_process(shared_lock, shared_check_dict, logger_name, gpu, is_hp_search, train_info_list, debug):
            mp_logger = make_logger.make(f"{logger_name}_gpu{gpu}", time_filename=False, save_dir=logger_dir)

            mp_st = time.time()
            mp_logger.info(f"train models with multiprocessing: gpus:{gpu}. start")

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu

            cached_data = CacheDataLoad(mp_logger)

            trained_list = []
            for train_info in train_info_list:
                log_prefix = train_info["log_prefix"]

                shared_lock.acquire()
                check_key = train_info["unique_key"]
                if check_key in shared_check_dict:
                    mp_logger.info(f"{log_prefix}. already started in other process\n")
                    shared_lock.release()
                    continue

                shared_check_dict[check_key] = True
                shared_lock.release()

                mp_logger.info(f"{log_prefix}. start")

                if debug:
                    # train(gpu=gpu, **train_info, folds=folds, logger=logger)
                    train(**train_info, cached_data=cached_data, folds=folds, is_hp_search=is_hp_search, logger=mp_logger, debug=debug)
                    trained_list.append(check_key)
                    continue

                try:
                    train(**train_info, cached_data=cached_data, folds=folds, is_hp_search=is_hp_search, logger=mp_logger, debug=debug)
                except Exception as e:
                    make_logger.write_exception_log(mp_logger, e)
                    exit(-1)

                trained_list.append(check_key)
                mp_logger.info("")

            mp_logger.info(f"trained list: total:{len(trained_list)}\n" + "\n".join(trained_list))
            mp_logger.info(f"train models with multiprocessing: gpus:{gpu}. end. elapse:{time.time()-mp_st}")

        process_list = []
        for gpu in gpu_devices:
            process = multiprocessing.Process(target=train_process, args=[shared_lock, shared_check_dict, logger_name, gpu, is_hp_search, train_info_list, debug])
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()

        logger.info("multiprocessing.end")
