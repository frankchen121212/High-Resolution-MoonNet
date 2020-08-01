import os
from lib.dataset import custom_image_generator
from keras.callbacks import EarlyStopping
from lib.evaluation import evaluation
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

    # Load checkpoint
    if args["checkpoint"]:
        init_weight_path = args["checkpoint"]
        if not os.path.exists(init_weight_path):
            raise Exception("Init Weight Path:{} Not Exists!".format(init_weight_path))
        logger.info("loading weight from {}".format(init_weight_path))
        model.load_weights(init_weight_path)

    best_F1socre = 0
    for epoch in range(args["epochs"]):
        n_samples = args["num_train"]
        batch_size = args["batch_size"]

        # Train
        model.fit_generator(
            custom_image_generator(Data['train'][0], Data['train'][1],
                                   batch_size=batch_size),
            steps_per_epoch= n_samples / batch_size, epochs=1, verbose=1,
            # validation_data=(Data['dev'][0],Data['dev'][1]), #no gen
            validation_data=custom_image_generator(Data['val'][0],
                                                   Data['val'][1],
                                                   batch_size=batch_size),
            validation_steps=n_samples,
            shuffle= True,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)]
        )

        # Debug prediction
        if args["debug_freq"] :
            debug_prediction(Data['val'],
                             model,
                             args["debug_freq"],
                             debug_path = os.path.join(output_path,'epoch_{}'.format(epoch)))


        # Evaluation
        avg_rec, avg_pre, avg_f1 = evaluation(Data['val'], Craters['val'], args["input_length"], model)
        logger.info("=> epoch {}/{}\t" .format(epoch,args["epochs"])\
                    + "rec:{}\tprec:{}\tf1_score:{}\t".format(avg_rec,
                                                             avg_pre,
                                                             avg_f1
                                                             )
                    )

        # saving checkpoint
        if avg_f1 > best_F1socre:
            best_F1socre = avg_f1
            model_name = get_model_file_name(args)
            Save_Model(model, model_name, logger)
