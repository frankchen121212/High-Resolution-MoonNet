import os
import numpy as np
from lib.dataset import custom_image_generator
from keras.callbacks import EarlyStopping
from lib.evaluation import evaluation,save_result
from lib.function import *
from lib.vis import debug_prediction

root = os.path.dirname(os.path.dirname(__file__))

def train(args, Data, Craters, model):

    # logger
    output_path = os.path.join(root, args["output_dir"], args["dataset"], args["model"])
    logger = log_creater(output_dir=output_path, mode='train')
    logger.info('==>Params')
    for key in args.keys():
        logger.info('\t{}:{}\n'.format(key, args[key]))
    log = get_log(args)

    # Load checkpoint
    if args["checkpoint"]:
        init_weight_path = args["checkpoint"]
        if not os.path.exists(init_weight_path):
            raise Exception("Init Weight Path:{} Not Exists!".format(init_weight_path))
        logger.info("loading weight from {}".format(init_weight_path))
        model.load_weights(init_weight_path)

    best_F1socre = 0
    min_loss = 9999
    for epoch in range(args["epochs"]):
        n_samples = args["num_train"]
        batch_size = args["batch_size"]

        # Train
        model.fit_generator(
            custom_image_generator(Data['train'][0], Data['train'][1],
                                   batch_size=batch_size),
            steps_per_epoch= n_samples / batch_size,
            epochs=args["epochs"],
            verbose=1,
            validation_data=custom_image_generator(Data['val'][0],
                                                   Data['val'][1],
                                                   batch_size=batch_size),
            validation_steps=n_samples,
            shuffle= True,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0),
                       log]
        )

        # Debug prediction
        if args["debug_freq"] :
            debug_prediction(Data['train'],
                             model,
                             args["debug_freq"],
                             debug_path = os.path.join(output_path,'epoch_train_{}'.format(epoch)))
            debug_prediction(Data['val'],
                             model,
                             args["debug_freq"],
                             debug_path=os.path.join(output_path, 'epoch_val_{}'.format(epoch)))

        # Evaluation


        # saving checkpoint
        if args["dataset"] == "deepmoon":
            recall, precision, fscore, _, _, _, _, _, _, _ = evaluation(Data['val'], Craters['val'],
                                                                        args["input_length"], model)
            avg_rec, avg_pre, avg_f1 = np.mean(recall), np.mean(precision), np.mean(fscore)

            logger.info("=> epoch {}/{}\t".format(epoch, args["epochs"]) \
                        + "rec:{}\tprec:{}\tf1_score:{}\t".format(avg_rec,
                                                                  avg_pre,
                                                                  avg_f1
                                                                  )
                        )
            # Save checkpoint
            if avg_f1 > best_F1socre:
                best_F1socre = avg_f1
                model_name = get_model_file_name(args)
                Save_Model(model, model_name, logger)
        elif args["dataset"] == "surfacecrack" or "assembled" in args["dataset"]:
            loss = model.evaluate(Data['val'][0], Data['val'][1])
            logger.info("=> epoch {}/{}\t".format(epoch, args["epochs"]) \
                        + "loss:{}\t".format(loss)
                        )
            # Save checkpoint
            if loss < min_loss:
                min_loss = loss
                model_name = get_model_file_name(args)
                Save_Model(model, model_name, logger)


def test(args, Data, Craters, model):
    # logger
    output_path = os.path.join(root, 'result', args["dataset"], args["model"])
    logger = log_creater(output_dir=output_path, mode='val')
    logger.info('==>Params')
    for key in args.keys():
        logger.info('\t{}:{}\n'.format(key, args[key]))

    # Load checkpoint
    if args["checkpoint"]:
        init_weight_path = args["checkpoint"]
        if not os.path.exists(init_weight_path):
            raise Exception("Init Weight Path:{} Not Exists!".format(init_weight_path))
        logger.info("loading weight from {}".format(init_weight_path))
        model.load_weights(init_weight_path)

    print("=> Evaluating Model.............")
    recall, precision, fscore,\
    frac_new, frac_new2, maxrad,\
    err_lo, err_la, err_r,\
    frac_duplicates = evaluation(Data['val'], Craters['val'], args["input_length"], model)

    save_result(logger, recall, precision, fscore,
                frac_new, frac_new2, maxrad,
                err_lo, err_la, err_r, frac_duplicates)