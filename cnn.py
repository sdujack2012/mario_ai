            layer_outputs = [layer for layer in (self.conv1,self.conv2,self.conv3)]
            activation_model = tf.keras.Model(inputs=self.input, outputs=layer_outputs)
            activations = activation_model.predict(input)
            
            activation = activations[2]
            activation_index=0
            fig, ax = plt.subplots(8, 8, figsize=(10,10))
            for row in range(0,8):
                for col in range(0,8):
                    ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
                    activation_index += 1    
            fig.savefig('test2png.png')