鸟类问题
鸟类鸣叫声的分析和分类是一个非常有趣的问题。
鸟类有许多类型的声音，而不同类型的声音具有不同的功能。最常见的是歌声和“其他声音”（如叫声）。
歌声是一种“美妙”的、有旋律的声音类型，鸟类通过它标记领地并吸引伴侣。它通常比“叫声”更复杂和更长。
叫声类型包括接触、引诱和警报声。接触和吸引叫声用于在飞行或觅食时将鸟类聚集在一起，例如在树冠中，警报声用于警示（例如当捕食者到来时）。这些叫声通常很短且简单。

为什么基于声音的鸟类分类会是一个具有挑战性的任务？
可能会遇到许多问题，包括：

背景噪声——特别是在使用城市录制的数据时（例如城市噪音、人声、汽车等） 

多标签分类问题——当同时有许多物种在鸣叫时

 不同类型的鸟类鸣叫声（如前面所述） 

物种间的变异性——同一物种在不同地区或国家可能存在鸟类鸣叫声的差异 

数据集问题——由于某些物种的流行度较高，数据可能不平衡，不同物种的录音长度、录音质量（音量、清晰度）也可能存在差异

那么过去是如何解决这些问题的呢？

通过鸟类声音识别来识别鸟类可能是一项困难的任务，但并不意味着不可能。但如何解决这些问题呢？
为了找到答案，需要深入研究论文，发现大部分工作都是由各种人工智能挑战赛发起的，比如BirdCLEF和DCASE。幸运的是，这些挑战的获胜者通常会描述他们的方法，所以在查看排行榜后，得到了一些有趣的见解：
几乎所有获胜的解决方案都使用了卷积神经网络（CNN）或循环卷积神经网络（RCNN）。 基于CNN的模型与浅层基于特征的方法之间的差距仍然相当大。 即使许多录音非常嘈杂，CNN仍然能够良好地工作，而不需要任何额外的降噪处理，许多团队声称降噪技术没有帮助。 数据增强技术似乎被广泛使用，特别是在音频处理中使用的时间或频率移位等技术。 一些获胜团队成功地采用了半监督学习方法（伪标签法），一些团队通过模型集成提高了AUC值。
但是，当我们只有声音录音时，如何应用于从图像中提取特征、进行分类或分割的神经网络设计的CNN呢？梅尔频率倒谱系数（MFCC）是答案。

SOUND_DIR='../data/xeno-canto-dataset-full/Parusmajor/Lithuania/Parusmajor182513.mp3'# Cargue el archivo mp3
signal, sr = librosa.load(SOUND_DIR,duration=10) # sr = sampling rate# Crea mel-espectrograma
N_FFT = 1024         
HOP_SIZE = 1024       
N_MELS = 128            
WIN_SIZE = 1024      
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'      
FMIN = 1400 S = librosa.feature.melspectrogram(y=signal,sr=sr,
                                    n_fft=N_FFT,
                                    hop_length=HOP_SIZE, 
                                    n_mels=N_MELS, 
                                    htk=True, 
                                    fmin=FMIN, 
                                    fmax=sr/2) plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S**2,ref=np.max), fmin=FMIN,y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.show()
 
梅尔尺度频谱图的示例
它是什么，它是如何工作的？
每个我们听到的声音都由多个声音频率同时组成。这就是使音频听起来“深沉”的原因。
声谱图的技巧是在一个图中可视化这些频率，而不仅仅像在波形中那样可视化振幅。梅尔频率刻度是似乎对听众来说彼此之间距离相等的音调音频刻度。这背后的想法与人类听觉方式有关。当我们将这两个想法结合起来，就得到了一个改进的声谱图（梅尔频率倒谱系数），它简单地忽略人类听不到的声音并绘制最重要的部分。
创建声谱图时，音频长度越长，你可以获得更多的图像信息，但模型可能过拟合的可能性也会增加。如果你的数据中有很多噪声或静音，那么5秒持续时间的音频可能无法捕捉到所需的信息。因此，决定将10秒的音频文件转换为图像（这样可以提高最终模型的准确性10%！）。由于鸟类在高频率上唱歌，因此应用了高通滤波器以去除无用的噪声。 
信息不足（静音）且以噪声为主的 5s 频谱图示例
建模
通过将数据分为训练集（80%）、验证集（10%）和测试集（10%），实现了对最终模型的训练。
IM_SIZE = (224,224,3) 
BIRDS = ['0Parus', '1Turdu', '2Passe', '3Lusci', '4Phoen', '5Erith',
'6Picap', '7Phoen', '8Garru', '9Passe', '10Cocco', '11Sitta','12Alaud', '13Strep', '14Phyll', '15Delic','16Turdu', '17Phyll','18Fring', '19Sturn', '20Ember', '21Colum', '22Trogl', '23Cardu','24Chlor', '25Motac', '26Turdu']
DATA_PATH = 'data/27_class_10s_2/'
BATCH_SIZE = 16

使用内置的Keras库数据生成器来处理所有声谱图的数据增强和归一化。

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,  
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.1,
                                   fill_mode='nearest')train_batches = train_datagen.flow_from_directory(DATA_PATH+'train',classes=BIRDS, target_size=IM_SIZE, class_mode='categorical', shuffle=True,batch_size=BATCH_SIZE)valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)valid_batches = valid_datagen.flow_from_directory(DATA_PATH+'val',classes=BIRDS,target_size=IM_SIZE, class_mode='categorical', shuffle=False, batch_size=BATCH_SIZE)test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)test_batches = test_datagen.flow_from_directory(DATA_PATH+'test', classes=BIRDS,target_size=IM_SIZE,class_mode='categorical', shuffle=False,batch_size=BATCH_SIZE)

最终的模型是基于EfficientNetB3构建的，包含27个不同的类（鸟类物种），使用Adam优化器，分类交叉熵损失函数和平衡的类权重。学习率会在平台上进行调整。

# Define la arquitectura de CNNnet = efn.EfficientNetB3(include_top=False,                       weights='imagenet', input_tensor=None,                        input_shape=IM_SIZE)x = net.output 
x = Flatten()(x) 
x = Dropout(0.5)(x)output_layer = Dense(len(BIRDS), activation='softmax', name='softmax')(x) 
net_final = Model(inputs=net.input, outputs=output_layer)      net_final.compile(optimizer=Adam(),                  loss='categorical_crossentropy', metrics=['accuracy'])# Estime los pesos de clase para el conjunto de datos no balanceadoclass_weights = class_weight.compute_class_weight(                'balanced',                 np.unique(train_batches.classes),                  train_batches.classes)# Define las callbacksModelCheck = ModelCheckpoint('models/efficientnet_checkpoint.h5', monitor='val_loss', verbose=0,                               save_best_only=True, save_weights_only=True, mode='auto', period=1)ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2,                               patience=5, min_lr=3e-4)
 
解决方案简介：音频数据预处理和神经网络模型
# Entrena el modelonet_final.fit_generator(train_batches,
                        validation_data = valid_batches,
                        epochs = 30,
                        steps_per_epoch= 1596,
                        class_weight=class_weights, callbacks[ModelCheck,ReduceLR])
模型的分类报告显示，在测试样本上，该解决方案以87%的准确率预测了正确的鸟类名称，其中：
11个类的F1分数超过90% 
8个类的F1分数介于70%和90%之间 
2个类的F1分数介于50%和70%之间 
6个类的F1分数低于50%。
 
神经网络模型分类报告
