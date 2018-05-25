


# framework

## Input_reader

supply data for network

## KittiDataset

handle dataset for kitti
do some aug_list

## Model
build model

## Train
train policy such as algorithm hyperparameters , save checkpoints

## Eval
eval policy, like Train








# TF Notes
## Moving Average
    ```
    opt = Optimizer(optimizer_config)
opt_op = opt.minimize(loss)
# shadow_variable -= (1 - decay) * (shadow_variable - variable)
    moving_average = tf.train.ExponentialMovingAverage(decay=0.99)
maintain_average_op = moving_average.apply([var0,var1])
    with tf.control_dependencies([opt_op]):

# inplace of the usual training op
        train_op = tf.group(maintain_average_op)
    ```

## Saver
    ```
saver = tf.train.Saver(var_list=None,max_to_keep,pad_step_number)

```

# load checkpoints
```
all_checkpoint_states = tf.train.get_checkpoint_state(checkpoint_dir)
    if all_checkpoint_states:
    all_checkpoint_paths =all_checkpoint_states.all_model_checkpoint_paths
    saver.recover_last_checkpoints(all_checkpoint_paths)
saver.restore(saver.last_checkpoints[-1])


saver.save(sess,save_path=checkpoint_path,global_step=global_step_scalar)
```

# Slim Note
 


## Summaries

```
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # protocol buffer
    summaries_op = tf.summary.merge(list(summaries),name='summaries_op')
```











# Python Notes
    * classmethod
    to create overload,pass cls automaticly
    * staticmethod
    function of checking something for class

