    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import ResNet50, ResNet101, DenseNet201, InceptionV3, VGG16, VGG19
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    import numpy as np
    import time
    from google.colab import drive

    drive.mount('/content/drive')

    # Define paths
    train_dir = '/content/drive/MyDrive/sem-4AIML/soil_detection/organized_dataset/train'
    validation_dir = '/content/drive/MyDrive/sem-4AIML/soil_detection/organized_dataset/validation'
    test_dir = '/content/drive/MyDrive/sem-4AIML/soil_detection/organized_dataset/test'

    # Create ImageDataGenerator instances
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load images
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical'
    )

    # Define a function to create the model
    def create_model(base_model_name):
        if base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
        elif base_model_name == 'ResNet101':
            base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
        elif base_model_name == 'DenseNet201':
            base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
        elif base_model_name == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
        elif base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
        elif base_model_name == 'VGG19':
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
        else:
            raise ValueError(f"Unsupported model name: {base_model_name}")

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(train_generator.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Compile the model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    # Define a function to evaluate the model
    def evaluate_model(model, model_name, test_generator):
        start_time = time.time()
        y_pred = model.predict(test_generator)
        prediction_time = time.time() - start_time

        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = test_generator.classes

        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

        if conf_matrix.size == 4:
            tn, fp, fn, tp = conf_matrix.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fpr = 0  # For multiclass, FPR calculation needs a different approach

        return {
            "Model Name": model_name,
            "Accuracy": accuracy * 100,
            "Precision": precision * 100,
            "Recall": recall * 100,
            "FPR": fpr * 100,
            "Prediction Time": prediction_time
        }

    model_names = ['ResNet50', 'ResNet101', 'DenseNet201', 'InceptionV3', 'VGG16', 'VGG19']

    for model_name in model_names:
        print(f"Training and evaluating model: {model_name}")
        model = create_model(model_name)

        history = model.fit(
            train_generator,
            epochs=10,  # Adjust epochs as needed
            validation_data=validation_generator
        )

        save_path = f'/content/drive/MyDrive/sem-4AIML/soil_detection/model/{model_name}_soil_classification_model.h5'
        model.save(save_path)
        print(f'Model saved to {save_path}')

        model = tf.keras.models.load_model(save_path)
        print(f'Model loaded from {save_path}')

        # Evaluate the model
        metrics = evaluate_model(model, model_name, test_generator)

        print(f"{'Model Name':<20} {'Accuracy (%)':<15} {'Precision (%)':<15} {'Recall (%)':<15} {'FPR (%)':<10} {'Prediction Time (s)':<20}")
        print(f"{metrics['Model Name']:<20} {metrics['Accuracy']:<15.2f} {metrics['Precision']:<15.2f} {metrics['Recall']:<15.2f} {metrics['FPR']:<10.2f} {metrics['Prediction Time']:<20.2f}")
        print("\n" + "="*80 + "\n")
