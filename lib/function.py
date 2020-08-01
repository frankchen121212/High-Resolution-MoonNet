import time, os,logging
root = os.path.dirname(os.path.dirname(__file__))

def log_creater(output_dir,mode = 'train'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = mode+'_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log

def Save_Model(model,model_name,logger):
    with open(model_name + ".json", 'w') as j_file:
        j_file.write(model.to_json())
    logger.info('saving =>{}'.format(model_name + ".json"))
    with open(model_name+".yaml", 'w') as y_file:
        y_file.write(model.to_yaml())
    logger.info('saving =>{}'.format(model_name+".yaml"))
    model.save_weights(model_name+".h5")
    logger.info('saving =>{}'.format(model_name+".h5"))

def get_model_file_name(args):
    model_file_name = (os.path.join(root,
                                    "model",
                                    args["dataset"],
                                    args["model"],
                                    args["output_prefix"]))
    if not os.path.exists(os.path.dirname(model_file_name)):
        os.makedirs(os.path.dirname(model_file_name))
    return model_file_name