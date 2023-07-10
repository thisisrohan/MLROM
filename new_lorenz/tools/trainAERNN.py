import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex":True,
    "font.family":"serif"
})


class NMSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, divisor_arr, name='NMSE', **kwargs):
        super(NMSE, self).__init__(name, **kwargs)
        self.divisor_arr = divisor_arr

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true / self.divisor_arr
        y_pred = y_pred / self.divisor_arr
        return super(NMSE, self).update_state(y_true, y_pred, sample_weight)

class NMSE_wt(tf.keras.metrics.MeanSquaredError):
    def __init__(self, divisor_arr, loss_weights=None, name='NMSE_wt', **kwargs):
        super(NMSE_wt, self).__init__(name, **kwargs)
        self.divisor_arr = divisor_arr
        self.loss_weights = loss_weights**0.5 if loss_weights is not None else loss_weights

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred / self.divisor_arr
        y_true = y_true / self.divisor_arr
        if isinstance(self.loss_weights, type(None)) == False:
            y_pred = y_pred * self.loss_weights
            y_true = y_true * self.loss_weights
        return super(NMSE_wt, self).update_state(y_true, y_pred, sample_weight)

class AERNN_loss(tf.keras.losses.Loss):
    def __init__(self, divisor_arr, loss_weights=None):
        super(AERNN_loss, self).__init__()
        self.divisor_arr = divisor_arr
        self.loss_weights = loss_weights**0.5 if loss_weights is not None else loss_weights

    def call(self, y_true, y_pred):
        y_pred = y_pred / self.divisor_arr
        y_true = y_true / self.divisor_arr
        if isinstance(self.loss_weights, type(None)) == False:
            y_pred = y_pred * self.loss_weights
            y_true = y_true * self.loss_weights
        return tf.keras.losses.MSE(y_pred, y_true)

def trainAERNN(
        create_data_for_RNN,
        Autoencoder,
        AR_RNN,
        all_data,
        AR_AERNN,
        **kwargs
    ):

    dt_rnn = kwargs['dt_rnn']
    T_sample_input = kwargs['T_sample_input']
    T_sample_output = kwargs['T_sample_output']
    T_offset = kwargs['T_offset']
    boundary_idx_arr = kwargs['boundary_idx_arr']
    delta_t = kwargs['delta_t']
    params = kwargs['params']
    normalize_dataset = kwargs['normalize_dataset']
    stddev_multiplier = kwargs['stddev_multiplier']
    skip_intermediate = kwargs['skip_intermediate']
    normalization_type = kwargs['normalization_type']
    normalization_constant_arr_aedata = kwargs['normalization_constant_arr_aedata']
    normalization_constant_arr_rnndata = kwargs['normalization_constant_arr_rnndata']
    learning_rate_list = kwargs['learning_rate_list']
    epochs = kwargs['epochs']
    patience = kwargs['patience']
    loss_weights = kwargs['loss_weights']
    min_delta = kwargs['min_delta']
    lambda_reg = kwargs['lambda_reg']
    stddev_rnn = kwargs['stddev_rnn']
    stateful = kwargs['stateful']
    behaviour = kwargs['behaviour']
    strategy = kwargs['strategy']
    dir_name_rnn = kwargs['dir_name_rnn']
    dir_name_AR_AErnn = kwargs['dir_name_AR_AErnn']
    batch_size = kwargs['batch_size']
    load_file_rnn = kwargs['load_file_rnn']
    wt_file_rnn = kwargs['wt_file_rnn']
    load_file_ae = kwargs['load_file_ae']
    wt_file_ae = kwargs['wt_file_ae']
    covmat_lmda = kwargs['covmat_lmda']
    readAndReturnLossHistories = kwargs['readAndReturnLossHistories']
    mytimecallback = kwargs['mytimecallback']
    plot_losses = kwargs['plot_losses']
    SaveLosses = kwargs['SaveLosses']
    train_split = kwargs['train_split']
    test_split = kwargs['test_split']
    val_split = kwargs['val_split']
    freeze_layers = kwargs.pop('freeze_layers', [])
    clipnorm = kwargs.pop('clipnorm', None)
    global_clipnorm = kwargs.pop('global_clipnorm', None)
    ESN_flag = kwargs.pop('ESN_flag', False)
    rnn_kwargs = kwargs.pop('rnn_kwargs', {})
    use_ae_data = kwargs.pop('use_ae_data', True)

    ### create autoencoder and load weights
    if use_ae_data == True:
        ae_net = Autoencoder(all_data.shape[1], load_file=load_file_ae)
        ae_net.load_weights_from_file(wt_file_ae)
    else:
        ae_net = None
        normalization_constant_arr_aedata = normalization_constant_arr_rnndata
        normalization_constant_arr_rnndata = None

    ### Creating data for AE-RNN and doing some statistics
    time_mean_ogdata = np.mean(all_data, axis=0)
    time_stddev_ogdata = np.std(all_data, axis=0)

    num_outsteps = int((T_sample_output + 0.5*dt_rnn)//dt_rnn)

    rnn_res_dict = create_data_for_RNN(
        all_data,
        dt_rnn,
        T_sample_input,
        T_sample_output,
        T_offset,
        None,
        boundary_idx_arr,
        delta_t,
        params=params,
        return_numsamples=True,
        normalize_dataset=normalize_dataset,
        stddev_multiplier=stddev_multiplier,
        skip_intermediate=skip_intermediate,
        return_OrgDataIdxArr=False,
        normalization_arr_external=normalization_constant_arr_aedata,
        normalization_type=normalization_type)
        
    data_rnn_input = rnn_res_dict['data_rnn_input']
    data_rnn_output = rnn_res_dict['data_rnn_output']
    org_data_idx_arr_input = rnn_res_dict['org_data_idx_arr_input']
    org_data_idx_arr_output = rnn_res_dict['org_data_idx_arr_output']
    num_samples = rnn_res_dict['num_samples']
    normalization_arr = rnn_res_dict['normalization_arr']
    rnn_data_boundary_idx_arr = rnn_res_dict['rnn_data_boundary_idx_arr']
    
    temp = np.divide(all_data-normalization_arr[0], normalization_arr[1])
    time_stddev = np.std(temp, axis=0)
    timeMeanofSpaceRMS = np.mean(np.mean(temp**2, axis=1)**0.5)
    del(org_data_idx_arr_input)
    del(org_data_idx_arr_output)
    del(temp)
    
    # if loss_weights is None:
    #     loss_weights = [[1.0]*data_rnn_output.shape[1]]
    # el
    if isinstance(loss_weights, list) == False:
        loss_weights = np.array([loss_weights**np.arange(data_rnn_output.shape[1])])
        loss_weights = np.tile(loss_weights, [data_rnn_output.shape[-1], 1])
        loss_weights = np.transpose(loss_weights)
    # print('loss_weights : ', loss_weights)

    ### splitting the data for train-val-test, and shuffling
    cum_samples = rnn_data_boundary_idx_arr[-1]
    # idx = np.arange(cum_samples)
    # np.random.shuffle(idx)
    num_train_arr = np.zeros(shape=rnn_data_boundary_idx_arr.shape[0], dtype='int32')
    num_val_arr = np.zeros(shape=rnn_data_boundary_idx_arr.shape[0], dtype='int32')
    num_test_arr = np.zeros(shape=rnn_data_boundary_idx_arr.shape[0], dtype='int32')
    num_samples_arr = np.zeros(shape=rnn_data_boundary_idx_arr.shape[0], dtype='int32')
    begin_idx = 0
    for i in range(len(rnn_data_boundary_idx_arr)):
        num_samples = rnn_data_boundary_idx_arr[i] - begin_idx
        num_train_arr[i] = batch_size * (int( (1-test_split-val_split)*num_samples )//batch_size)
        num_val_arr[i] = batch_size * (int(val_split*num_samples)//batch_size)
        num_test_arr[i] = batch_size * int((num_samples - num_train_arr[i] - num_val_arr[i])//batch_size)
        num_samples_arr[i] = num_train_arr[i] + num_val_arr[i] + num_test_arr[i]
        begin_idx = rnn_data_boundary_idx_arr[i]

    # defining shapes
    training_input_shape = [np.sum(num_train_arr)]
    training_input_shape.extend(data_rnn_input.shape[1:])

    training_output_shape = [np.sum(num_train_arr)]
    training_output_shape.extend(data_rnn_output.shape[1:])

    val_input_shape = [np.sum(num_val_arr)]
    val_input_shape.extend(data_rnn_input.shape[1:])

    val_output_shape = [np.sum(num_val_arr)]
    val_output_shape.extend(data_rnn_output.shape[1:])

    testing_input_shape = [np.sum(num_test_arr)]
    testing_input_shape.extend(data_rnn_input.shape[1:])

    testing_output_shape = [np.sum(num_test_arr)]
    testing_output_shape.extend(data_rnn_output.shape[1:])

    # defining required arrays
    training_data_rnn_input = np.empty(shape=training_input_shape)
    training_data_rnn_output = np.empty(shape=training_output_shape)

    val_data_rnn_input = np.empty(shape=val_input_shape)
    val_data_rnn_output = np.empty(shape=val_output_shape)

    testing_data_rnn_input = np.empty(shape=testing_input_shape)
    testing_data_rnn_output = np.empty(shape=testing_output_shape)

    begin_idx = 0
    training_data_rolling_count = 0
    val_data_rolling_count = 0
    testing_data_rolling_count = 0
    for i in range(len(rnn_data_boundary_idx_arr)):
        idx = np.arange(begin_idx, rnn_data_boundary_idx_arr[i])
        num_samples = num_samples_arr[i]
        num_train = num_train_arr[i]
        num_val = num_val_arr[i]
        num_test = num_test_arr[i]

        nbatches_train = num_train // batch_size
        nbatches_val = num_val // batch_size
        nbatches_test = num_test // batch_size

        for j in range(batch_size):
            training_data_rnn_input[training_data_rolling_count+j:training_data_rolling_count+num_train:batch_size] = data_rnn_input[idx[0:num_train]][j*nbatches_train:(j+1)*nbatches_train]
            training_data_rnn_output[training_data_rolling_count+j:training_data_rolling_count+num_train:batch_size] = data_rnn_output[idx[0:num_train]][j*nbatches_train:(j+1)*nbatches_train]
            
            val_data_rnn_input[val_data_rolling_count+j:val_data_rolling_count+num_val:batch_size] = data_rnn_input[idx[num_train:num_train+num_val]][j*nbatches_val:(j+1)*nbatches_val]
            val_data_rnn_output[val_data_rolling_count+j:val_data_rolling_count+num_val:batch_size] = data_rnn_output[idx[num_train:num_train+num_val]][j*nbatches_val:(j+1)*nbatches_val]

            testing_data_rnn_input[testing_data_rolling_count+j:testing_data_rolling_count+num_test:batch_size] = data_rnn_input[idx[num_train+num_val:num_samples]][j*nbatches_test:(j+1)*nbatches_test]
            testing_data_rnn_output[testing_data_rolling_count+j:testing_data_rolling_count+num_test:batch_size] = data_rnn_output[idx[num_train+num_val:num_samples]][j*nbatches_test:(j+1)*nbatches_test]

        training_data_rolling_count += num_train

        val_data_rolling_count += num_val

        testing_data_rolling_count += num_test

        begin_idx = rnn_data_boundary_idx_arr[i]

    # cleaning up
    del(data_rnn_input)
    del(data_rnn_output)

    # further shuffling
    if stateful == False:
        idx = np.arange(0, training_data_rnn_input.shape[0])
        np.random.shuffle(idx)
        training_data_rnn_input = training_data_rnn_input[idx]
        training_data_rnn_output = training_data_rnn_output[idx]

        idx = np.arange(0, val_data_rnn_input.shape[0])
        np.random.shuffle(idx)
        val_data_rnn_input = val_data_rnn_input[idx]
        val_data_rnn_output = val_data_rnn_output[idx]

        idx = np.arange(0, testing_data_rnn_input.shape[0])
        np.random.shuffle(idx)
        testing_data_rnn_input = testing_data_rnn_input[idx]
        testing_data_rnn_output = testing_data_rnn_output[idx]

        del(idx)
        
        
    ### Initialize RNN network
    if strategy is not None:
        with strategy.scope():
            rnn_net = AR_RNN(
                load_file=load_file_rnn,
                T_input=T_sample_input,
                T_output=T_sample_output,
                stddev=stddev_rnn,
                batch_size=batch_size,
                lambda_reg=lambda_reg,
                # stateful=stateful,
                **rnn_kwargs,
            )
    else:
        rnn_net = AR_RNN(
            load_file=load_file_rnn,
            T_input=T_sample_input,
            T_output=T_sample_output,
            stddev=stddev_rnn,
            batch_size=batch_size,
            lambda_reg=lambda_reg,
            # stateful=stateful,
            **rnn_kwargs,
        )
    save_path = dir_name_AR_AErnn+'/rnn_net'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if behaviour == 'initialiseAndTrainFromScratch':
        rnn_net.save_class_dict(save_path+'/final_net_class_dict.txt')
        rnn_net.build(input_shape=(batch_size, training_data_rnn_input.shape[1], rnn_net.data_dim))        
    rnn_net.load_weights_from_file(wt_file_rnn)
    
    for i in range(len(freeze_layers)):
        rnn_net.rnn_list[freeze_layers[i]].trainable = False
    
    # compiling the RNN network
    rnn_net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_list[0]),
        loss=tf.keras.losses.MeanSquaredError(),
        run_eagerly=False,
        # loss_weights=loss_weights,
        metrics='mse'
    )

    if behaviour == 'loadCheckpointAndContinueTraining':# and ESN_flag == False:
        # this loads the weights/attributes of the optimizer as well
        wt_file = tf.train.latest_checkpoint(dir_name_AR_AErnn+'/checkpoints')
        if strategy is not None:
            with strategy.scope():
                rnn_net.load_weights(wt_file)
        else:
            rnn_net.load_weights(wt_file)


    ### AE-RNN
    AR_AERNN_net = AR_AERNN(
        ae_net,
        rnn_net,
        normalization_constant_arr_rnndata,
        normalization_constant_arr_aedata,
        covmat_lmda,
        time_stddev_ogdata,
        time_mean_ogdata,
        loss_weights=loss_weights,
        clipnorm=clipnorm,
        global_clipnorm=global_clipnorm,
    )

    AR_AERNN_net.build(input_shape=(batch_size,)+training_data_rnn_input.shape[1:])

    
    AR_AERNN_net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_list[0]),# clipnorm=clipnorm),
        loss=AERNN_loss(divisor_arr=time_stddev, loss_weights=loss_weights),#_ogdata/normalization_constant_arr_aedata[1, :]),
        run_eagerly=False,
        # loss_weights=loss_weights,
        metrics=['mse', NMSE(divisor_arr=time_stddev)],
    )
    
    if behaviour == 'initialiseAndTrainFromScratch':
        val_loss_hist = []
        train_loss_hist = []
        lr_change=[0, 0]
        arr_len = 0
        if type(epochs) != type([]):
            arr_len = epochs*len(learning_rate_list)
        else:
            for i in range(len(epochs)):
                arr_len += epochs[i]
        savelosses_cb_vallossarr = np.ones(shape=arr_len)*np.NaN
        savelosses_cb_trainlossarr = np.ones(shape=arr_len)*np.NaN
        starting_lr_idx = 0
        num_epochs_left = epochs if type(epochs) != type([]) else [ep for ep in epochs]
        earlystopping_wait = 0
    elif behaviour == 'loadCheckpointAndContinueTraining':
        if type(epochs) != type([]):
            arr_len = epochs*len(learning_rate_list)
        else:
            arr_len = 0
            for i in range(len(epochs)):
                arr_len += epochs[i]
        val_loss_hist, train_loss_hist, lr_change, starting_lr_idx, num_epochs_left, val_loss_arr_fromckpt, train_loss_arr_fromckpt, earlystopping_wait = readAndReturnLossHistories(
            dir_name_ae=dir_name_AR_AErnn,
            dir_sep='/',
            epochs=epochs,
            learning_rate_list=learning_rate_list,
            return_earlystopping_wait=True,
            fname='LossHistoriesCheckpoint-{}_outsteps.hdf5'.format(num_outsteps))
        savelosses_cb_vallossarr = val_loss_arr_fromckpt
        savelosses_cb_trainlossarr = train_loss_arr_fromckpt
    elif behaviour == 'loadFinalNetAndPlot':
        with open(dir_name_AR_AErnn+'/final_net/losses-{}_outsteps.txt'.format(num_outsteps)) as f:
            lines = f.readlines()
        
        losses_dict = eval(''.join(lines))

        val_loss_hist = losses_dict['val_loss_hist']
        train_loss_hist = losses_dict['train_loss_hist']
        lr_change = losses_dict['lr_change']
        test_loss = losses_dict['test_loss']

    train_NMSE_hist = []
    val_NMSE_hist = []
    
    train_MSE_hist = []
    val_MSE_hist = []
    
    train_global_gradnorm_hist = []
    train_covmat_fro_loss_hist = []
    
    rho_res_hist = []
    alpha_hist = []
    omega_in_hist = []

    ### training AR AE-RNN
    if behaviour == 'initialiseAndTrainFromScratch' or behaviour == 'loadCheckpointAndContinueTraining':
        # implementing early stopping
        baseline = None
        if behaviour == 'loadCheckpointAndContinueTraining':
            baseline = np.min(val_loss_hist)
        else:
            baseline = AR_AERNN_net.evaluate(
                val_data_rnn_input, val_data_rnn_output,
            )
            baseline = baseline[2] # val_NMSE of RNN with loaded weights
            print('baseline : {:.4E}'.format(baseline))

        # time callback for each epoch
        timekeeper_cb = mytimecallback()

        # model checkpoint callback
        dir_name_ckpt = dir_name_AR_AErnn+'/checkpoints'
        if not os.path.isdir(dir_name_ckpt):
            os.makedirs(dir_name_ckpt)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=dir_name_ckpt+'/checkpoint-{}_outsteps'.format(num_outsteps),#+'/checkpoint--loss={loss:.4f}--vall_loss={val_loss:.4f}',
            monitor='val_NMSE',
            save_best_only=True,
            save_weights_only=True,
            verbose=2,
            initial_value_threshold=baseline,
            period=1  # saves every `period` epochs
        )

        epochs_so_far = 0
        for i in range(0, starting_lr_idx):
            if type(epochs) == type(list):
                if i < len(epochs):
                    epochs_i = epochs[i]
                epochs_so_far += epochs_i
            else:
                epochs_so_far += epochs

        for i in range(starting_lr_idx, len(learning_rate_list)):
            learning_rate = learning_rate_list[i]
            if type(patience) == type([]):
                if i < len(patience):
                    patience_thislr = patience[i]
                else:
                    patience_thislr = patience[-1]
            else:
                patience_thislr = patience
            if type(epochs) == type([]):
                if i < len(epochs):
                    epochs_thislr = epochs[i]
                else:
                    epochs_thislr = epochs[-1]
            else:
                epochs_thislr = epochs
                

            tf.keras.backend.set_value(AR_AERNN_net.optimizer.lr, learning_rate)

            # save losses callback
            savelosses_cb = SaveLosses(
                filepath=dir_name_ckpt+'/LossHistoriesCheckpoint-{}_outsteps'.format(num_outsteps),
                val_loss_arr=savelosses_cb_vallossarr,
                train_loss_arr=savelosses_cb_trainlossarr,
                total_epochs=epochs_thislr,
                period=1)
            # savelosses_cb.update_lr_idx(i)
            
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                monitor='val_NMSE',
                patience=patience_thislr,
                restore_best_weights=True,
                verbose=True,
                min_delta=min_delta,
                baseline=baseline
            )
            #** the two lines below are useless because wait is set to 0 in on_train_begin
            # early_stopping_cb.wait = earlystopping_wait
            # print('early_stopping_cb.wait : {}\n'.format(early_stopping_cb.wait))

            if i == starting_lr_idx:
                num_epochs_left_i = num_epochs_left[i] if type(num_epochs_left) == type([]) else num_epochs_left
                EPOCHS = num_epochs_left_i
                savelosses_cb.update_offset(epochs_so_far + epochs_thislr-num_epochs_left_i)
            else:
                EPOCHS = epochs_thislr
                savelosses_cb.update_offset(epochs_so_far + 0)

            total_s_len = 80
            sep_lr_s = ' LEARNING RATE : {} '.format(learning_rate)
            sep_lr_s = int((total_s_len - len(sep_lr_s))//2)*'-' + sep_lr_s
            sep_lr_s = sep_lr_s + (total_s_len-len(sep_lr_s))*'-'
            print('\n\n' + '-'*len(sep_lr_s))
            print('\n' + sep_lr_s+'\n')
            print('-'*len(sep_lr_s) + '\n\n')
            
            history = AR_AERNN_net.fit(training_data_rnn_input, training_data_rnn_output,
                epochs=EPOCHS,
                batch_size=batch_size,
    #             validation_split=val_split/train_split,
                validation_data=(val_data_rnn_input, val_data_rnn_output),
                callbacks=[early_stopping_cb, timekeeper_cb, checkpoint_cb, savelosses_cb],
                verbose=1,
                shuffle=True,
            )

            val_loss_hist.extend(history.history['val_loss'])
            train_loss_hist.extend(history.history['loss'])
            
            val_NMSE_hist.extend(history.history['val_NMSE'])
            train_NMSE_hist.extend(history.history['NMSE'])
            
            val_MSE_hist.extend(history.history['val_mse'])
            train_MSE_hist.extend(history.history['mse'])
            
            train_global_gradnorm_hist.extend(history.history['global_gradnorm'])
            train_covmat_fro_loss_hist.extend(history.history['covmat_fro_loss'])
            
            if 'rho_res_0' in history.history.keys():
                temp = []
                for jj in range(len(AR_AERNN_net.rnn_net.rnn_list)):
                    temp.append(history.history['rho_res_{}'.format(jj)])
                rho_res_hist.append(temp)
            if 'alpha_0' in history.history.keys():
                temp = []
                for jj in range(len(AR_AERNN_net.rnn_net.rnn_list)):
                    temp.append(history.history['alpha_{}'.format(jj)])
                alpha_hist.append(temp)
            if 'omega_in_0' in history.history.keys():
                temp = []
                for jj in range(len(AR_AERNN_net.rnn_net.rnn_list)):
                    temp.append(history.history['omega_in_{}'.format(jj)])
                omega_in_hist.append(temp)

            if i == starting_lr_idx:
                lr_change[i+1] += len(history.history['val_loss'])
            else:
                lr_change.append(lr_change[i]+len(history.history['val_loss']))
            
            epochs_so_far += epochs_thislr
            
    if behaviour == 'initialiseAndTrainFromScratch' or behaviour == 'loadCheckpointAndContinueTraining':
        # test_loss = rnn_net.evaluate(
        #     testing_data_rnn_input, testing_data_rnn_output,
        # )
        for layer in AR_AERNN_net.rnn_net.rnn_list:
            if layer.stateful == True:
                layer.reset_states()
        # print(testing_data_rnn_input.shape, testing_data_rnn_output.shape)
        
        test_mse = 0.0
        for i in range(int(testing_data_rnn_input.shape[0]//batch_size)):
            # i_test_loss = rnn_net.evaluate(
            #     testing_data_rnn_input[i*batch_size:(i+1)*batch_size, :, :],
            #     testing_data_rnn_output[i*batch_size:(i+1)*batch_size, :, :],
            # )
            data_in_i = testing_data_rnn_input[i*batch_size:(i+1)*batch_size, :, :]
            data_out_i = testing_data_rnn_output[i*batch_size:(i+1)*batch_size, :, :]
            temp = AR_AERNN_net.call(data_in_i, training=False)
            i_test_mse = np.mean(
                (
                    (data_out_i - temp.numpy()) * normalization_constant_arr_aedata[1, :] / time_stddev_ogdata
                )**2
            )
            test_mse = (i*test_mse + i_test_mse)/(i+1)

        save_path = dir_name_AR_AErnn+'/final_net'

        if not os.path.isdir(save_path):
            os.makedirs(save_path)


        with open(save_path+'/losses-{}_outsteps.txt'.format(num_outsteps), 'w') as f:
            f.write(str({
                'lr_change':lr_change,
                'test_mse':test_mse,
                'clipnorm':clipnorm,
                'global_clipnorm':global_clipnorm,
                'val_loss_hist':val_loss_hist,
                'train_loss_hist':train_loss_hist,
                'val_MSE_hist':val_MSE_hist,
                'train_MSE_hist':train_MSE_hist,
                'val_NMSE_hist':val_NMSE_hist,
                'train_NMSE_hist':train_NMSE_hist,
                'train_global_gradnorm_hist':train_global_gradnorm_hist,
                'train_covmat_fro_loss_hist':train_covmat_fro_loss_hist,
                'rho_res_hist':rho_res_hist,
                'alpha_hist':alpha_hist,
                'omega_in_hist':omega_in_hist,
            }))
            
        if normalize_dataset == True:
            with open(save_path+'/rnn_normalization.txt', 'w') as f:
                f.write(str({
                    'normalization_arr':normalization_arr
                }))

        AR_AERNN_net.save_everything(
            file_name=save_path+'/final_net-{}_outsteps'.format(num_outsteps)
        )
            
    # plotting losses
    dir_name_plot = dir_name_AR_AErnn+'/plots'
    if not os.path.isdir(dir_name_plot):
        os.makedirs(dir_name_plot)

    # Visualize loss history
    fig, ax = plot_losses(
        training_loss=train_loss_hist,
        val_loss=val_loss_hist,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
    )
    plt.savefig(dir_name_plot+'/loss_history-{}_outsteps.pdf'.format(num_outsteps), dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
    
    
    fig, ax = plot_losses(
        training_loss=train_MSE_hist,
        val_loss=val_MSE_hist,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
        legend_list=['Training MSE', 'Validation MSE'],
        xlabel='Epoch',
        ylabel='MSE',
    )
    plt.savefig(dir_name_plot+'/MSE_history-{}_outsteps.pdf'.format(num_outsteps), dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
    
    fig, ax = plot_losses(
        training_loss=train_NMSE_hist,
        val_loss=val_NMSE_hist,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
        legend_list=['Training NMSE', 'Validation NMSE'],
        xlabel='Epoch',
        ylabel='NMSE',
    )
    plt.savefig(dir_name_plot+'/NMSE_history-{}_outsteps.pdf'.format(num_outsteps), dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
    
    fig, ax = plot_losses(
        training_loss=train_global_gradnorm_hist,
        val_loss=None,
        lr_change=lr_change,
        learning_rate_list=learning_rate_list,
        legend_list=['global_gradnorm'],
        xlabel='Epoch',
        ylabel='gradnorm',
        plot_type='plot',
    )
    plt.savefig(dir_name_plot+'/train_global_gradnorm_history-{}_outsteps.pdf'.format(num_outsteps), dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

    if len(rho_res_hist) > 0:
        temp = [[elem for elem in lst] for lst in rho_res_hist[0]]
        for i in range(1, len(rho_res_hist)):
            for j in range(len(temp)):
                temp[j].extend(rho_res_hist[i][j])
                
        arr1 = temp[0]
        arr2 = None
        if len(temp) > 1:
            arr2 = temp[1]
        remaining_arrs = []
        more_plot_arrs_kwargs = []
        if len(temp) > 2:
            remaining_arrs = temp[2:]
            more_plot_arrs_kwargs = [{'linewidth':0.9}]*(len(temp) - 2)
        fig, ax = plot_losses(
            training_loss=arr1,
            val_loss=arr2,
            more_plot_arrs_lst=remaining_arrs,
            lr_change=lr_change,
            learning_rate_list=learning_rate_list,
            legend_list=['Layer {}'.format(i+1) for i in range(len(AR_AERNN_net.rnn_net.rnn_list))],
            xlabel='Epoch',
            ylabel=r'$\rho$',
            traininglossplot_args=[],
            traininglossplot_kwargs={'linewidth':0.9},
            vallossplot_args=[],
            vallossplot_kwargs={'linewidth':0.9},
            more_plot_arrs_args=[],
            more_plot_arrs_kwargs=more_plot_arrs_kwargs,
            plot_type='plot',
        )
        plt.savefig(dir_name_plot+'/rho_res_history-{}_outsteps.pdf'.format(num_outsteps), dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()

    if len(alpha_hist) > 0:
        temp = [[elem for elem in lst] for lst in alpha_hist[0]]
        for i in range(1, len(alpha_hist)):
            for j in range(len(temp)):
                temp[j].extend(alpha_hist[i][j])
                
        arr1 = temp[0]
        arr2 = None
        if len(temp) > 1:
            arr2 = temp[1]
        remaining_arrs = []
        more_plot_arrs_kwargs = []
        if len(temp) > 2:
            remaining_arrs = temp[2:]
            more_plot_arrs_kwargs = [{'linewidth':0.9}]*(len(temp) - 2)
        fig, ax = plot_losses(
            training_loss=arr1,
            val_loss=arr2,
            more_plot_arrs_lst=remaining_arrs,
            lr_change=lr_change,
            learning_rate_list=learning_rate_list,
            legend_list=['Layer {}'.format(i+1) for i in range(len(AR_AERNN_net.rnn_net.rnn_list))],
            xlabel='Epoch',
            ylabel=r'$\alpha$',
            traininglossplot_args=[],
            traininglossplot_kwargs={'linewidth':0.9},
            vallossplot_args=[],
            vallossplot_kwargs={'linewidth':0.9},
            more_plot_arrs_args=[],
            more_plot_arrs_kwargs=more_plot_arrs_kwargs,
            plot_type='plot',
        )
        plt.savefig(dir_name_plot+'/alpha_history-{}_outsteps.pdf'.format(num_outsteps), dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()

    if len(omega_in_hist) > 0:
        temp = [[elem for elem in lst] for lst in omega_in_hist[0]]
        for i in range(1, len(omega_in_hist)):
            for j in range(len(temp)):
                temp[j].extend(omega_in_hist[i][j])
                
        arr1 = temp[0]
        arr2 = None
        if len(temp) > 1:
            arr2 = temp[1]
        remaining_arrs = []
        more_plot_arrs_kwargs = []
        if len(temp) > 2:
            remaining_arrs = temp[2:]
            more_plot_arrs_kwargs = [{'linewidth':0.9}]*(len(temp) - 2)
        fig, ax = plot_losses(
            training_loss=arr1,
            val_loss=arr2,
            more_plot_arrs_lst=remaining_arrs,
            lr_change=lr_change,
            learning_rate_list=learning_rate_list,
            legend_list=['Layer {}'.format(i+1) for i in range(len(AR_AERNN_net.rnn_net.rnn_list))],
            xlabel='Epoch',
            ylabel=r'$\Omega_{in}$',
            traininglossplot_args=[],
            traininglossplot_kwargs={'linewidth':0.9},
            vallossplot_args=[],
            vallossplot_kwargs={'linewidth':0.9},
            more_plot_arrs_args=[],
            more_plot_arrs_kwargs=more_plot_arrs_kwargs,
            plot_type='plot',
        )
        plt.savefig(dir_name_plot+'/omega_in_history-{}_outsteps.pdf'.format(num_outsteps), dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()
 

    ### cleaning up
    del(training_data_rnn_input)
    del(training_data_rnn_output)
    del(val_data_rnn_input)
    del(val_data_rnn_output)
    del(testing_data_rnn_input)
    del(testing_data_rnn_output)

    return
        
