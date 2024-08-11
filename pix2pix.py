import tensorflow as tf
import args # 👈 配置所有的参数
import matplotlib.pyplot as plt


# 抽象出 pix 2 pix 类
class pix2pix(tf.keras.Model):
    def __init__(self,generator,discriminator):
        super(pix2pix, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gp_weight = args.gp_weight

    # 定义好优化器跟损失函数 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
    def compile(self,gen_optimizer,g_loss_fn,d_optimizer,d_loss_fn):
        super().compile()
        self.gen_optmizer = gen_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_optimizer = d_optimizer
        self.d_loss_fn = d_loss_fn
        self.idloss_fn = tf.keras.losses.MeanAbsoluteError()
        self.idweight = args.id_weight

    # 计算梯度 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
    def gradient_penalty_g(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.generator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def gradient_penalty_d(self, batch_size, real_images,x_train,fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated,x_train], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    # 🔥🔥🔥 训练判别器 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
    def train_step(self, data):
        x_train,y_train = args.get_x_y_train(data) # 👈 预处理数据的方法(因为输入的 data 图片是合并在一起的),方法也可配置, 写在 args.py 内

        # train D
        for i in range(args.d_step):
            with tf.GradientTape() as tape:
                # 生成原始假图
                fake_images = self.generator(x_train,training=True)
                # 进行判别
                fake_preds = self.discriminator([fake_images,x_train],training=True) # 传入假图跟判别的图（条件图 x_train）
                # 传入真图进行判别
                real_preds = self.discriminator([y_train,x_train],training=True)
                # 对比真假图的损失值
                d_cost = self.d_loss_fn(real_preds,fake_preds)
                gp_d = self.gradient_penalty_d(args.batch_size,y_train,x_train,fake_images)
                # Add the gradient penalty to the original discriminator loss
                # 得到总的 loss 值
                d_loss = d_cost + gp_d * self.gp_weight
                
            # 更新参数
            d_grads = tape.gradient(d_loss,self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_grads,self.discriminator.trainable_weights))

        # train G
        for i in range(args.g_step):
            with tf.GradientTape() as tape:
                fake_images = self.generator(x_train,training=True)
                preds = self.discriminator([fake_images,x_train],training=True)
                gp_g = self.gradient_penalty_g(args.batch_size,y_train,fake_images)
                g_loss = self.g_loss_fn(preds) + args.id_weight * self.idloss_fn(y_train,fake_images) + self.gp_weight * gp_g
            g_grads = tape.gradient(g_loss,self.generator.trainable_weights)
            self.gen_optmizer.apply_gradients(zip(g_grads,self.generator.trainable_weights))

        return {
            "g_loss":g_loss,
            "d_loss":d_loss
        }

    def call(self, inputs):
        self.gen_images = self.generator(self.inputs)

        return self.gen_images