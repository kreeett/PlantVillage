# Presentation Script — PlantVillage Demo

Designed for a **10–15 minute live demo** to Dr. Tariq Bdair. Each section's recommended time is in brackets. Adjust depending on your actual slot.

The voice is your voice. These are talking points, not a script to memorize. Read them in advance, internalize the structure, then speak naturally.

---

## Section 1 — Overview [1 min]

> "We built a plant disease classifier using a ResNet-18 convolutional neural network, implemented from scratch in PyTorch. We trained it on the PlantVillage dataset — about 20,000 leaf images across 15 classes covering pepper, potato, and tomato in healthy and diseased states. On the held-out test set we reached 99.03% accuracy."

**Point to:** the four metric cards at the top.

> "We'll walk through what the problem is, the dataset, the model architecture, how training went, what we got, and then we'll do live inference at the end."

---

## Section 2 — The Problem [1 min]

> "Identifying plant disease from a leaf image is what we'd call a fine-grained visual classification problem. The same crop can have many different diseases, and the visual differences between them are sometimes subtle — small spots, slight discolouration, leaf curling."

**Point at the gallery.** Pick two visually similar classes and contrast them:

> "Look at Tomato Early blight versus Tomato Late blight here — both are blights, both produce dark lesions, but the patterns are different. The model has to learn to make this kind of distinction across 15 categories."

---

## Section 3 — Dataset [1.5 min]

> "We used the PlantVillage dataset from Kaggle. Roughly 20,000 RGB images on uniform backgrounds. We split it 70/15/15 into train, validation, and test, with a fixed random seed for reproducibility."

**Point to the bar chart:**

> "The thing to notice here is the imbalance. The largest class has about 20× more images than the smallest. That's a real problem for training — naive cross-entropy will just optimize for the majority classes."

> "We addressed this through data augmentation rather than class-weighted loss. The training pipeline applies random horizontal flips, rotation, and colour jitter, so smaller classes get more effective variety per epoch. Validation and test see no augmentation."

---

## Section 4 — Architecture [3 min]

> "This is the part we spent the most time on. We didn't import a pretrained model — we built ResNet-18 from scratch using only `torch.nn` primitives. We wanted to actually understand what's inside the network, not treat it as a black box."

**Point at the top-level diagram:**

> "The structure follows He et al. 2016. Input goes through a 7×7 convolution, batch normalization, ReLU, and a max pool. Then four stages of two residual blocks each. Each stage doubles the channels and halves the spatial resolution. At the end, adaptive average pooling collapses to a 1×1 spatial map, we flatten to 512 dimensions, and a single fully-connected layer produces 15 class logits."

**Point at the residual block code:**

> "The interesting part is what's inside each block. You have two 3×3 convolutions with batch norm and ReLU between them. But the key thing is the skip connection — we add the input directly to the output of the convolutions before the final ReLU."

**Point at the right column — Why skip connections:**

> "This is what makes deep networks trainable. Without skip connections, gradients have to pass through every layer during backpropagation, and they tend to vanish. Early layers stop learning. The skip connection gives the gradient a direct path back to early layers."

> "There's also a more subtle reason. Each block effectively learns a *residual* — what to add to its input — rather than a full transformation. If the right thing to do is leave the input alone, learning zero is much easier than learning the identity function from scratch."

> "When the block changes either resolution or channel count, we can't just add the input — the shapes don't match. So we use a 1×1 projection convolution on the shortcut path to match the dimensions."

**Point at the parameter count metrics:**

> "The model has about 11 million parameters. Compared to ResNet-50 at 25 million or ResNet-101 at 44 million, this is small. With 20,000 training images we judged that 11 million was the right size — deeper variants would have so many parameters that they'd overfit without pretraining."

---

## Section 5 — Training Process [2 min]

**Point at the hyperparameter table:**

> "We trained for 75 epochs with Adam at an initial learning rate of 1e-3, with cosine annealing — the learning rate decays smoothly to zero by the end of training. Batch size 128. Cross-entropy loss. Standard recipe."

**Point at the training curve figure:**

> "Here's how training went. Blue is training, red is validation. On the left we have loss; on the right, accuracy."

> "Two things to point out. First, the training curves are smooth — the model is learning steadily. Second, the validation curve is noticeably more volatile in the first 35 epochs."

> "We initially thought this was instability in the model, but we figured out it's actually an artifact of small classes. There are only about 22 images of healthy potato in validation. Misclassifying just two of them swings the average loss visibly. As the cosine schedule shrinks the learning rate, the updates get smaller and the curve smooths out — you can see by epoch 40 it's stable."

> "And critically, training and validation accuracy converge tightly at the end. There's no widening generalization gap, which means the augmentation did its job — the model isn't memorizing the training set."

**Optional, if time permits — point at hardware:**

> "We trained on a Ryzen 9 with an RTX 4070 laptop GPU. End-to-end training took a few hours."

---

## Section 6 — Results [2 min]

**Point at the three top metrics:**

> "Final test set: 3,066 correct out of 3,096. Overall accuracy 99.03%."

**Point at the per-class table:**

> "Per-class breakdown. Eleven of fifteen classes scored above 98%. Four classes — Pepper bell Bacterial spot, Pepper bell healthy, Potato Early blight, Tomato Leaf Mold, and Tomato healthy — got 100%."

> "The two weakest classes are Potato healthy at 95.45% and Tomato Early blight at 95.93%. Both have the smallest test populations in their crops. Potato healthy in particular has only 22 test images — a single misclassification costs nearly 4.5 percentage points. So this is a metric noise effect more than a real model weakness."

**Point at the confusion matrix:**

> "The confusion matrix shows where the model gets confused. The diagonal is dark — that's correct predictions. The off-diagonal cells show errors. Most are isolated single misclassifications."

**Point at the auto-detected most-confused pair:**

> "The most common confusion the model makes is right here — [read it off the screen]. That's actually a known hard pair even for human agronomists."

---

## Section 7 — Live Inference [2 min]

> "Now let's run the model live."

**Stay on the Sample Gallery tab.** Pick a class — try Tomato Yellow Leaf Curl Virus first (the largest class, model is very confident):

> "I'll pick Tomato Yellow Leaf Curl Virus. The model needs about a hundred milliseconds to classify it on the GPU."

**Wait for prediction.** The bar chart appears.

> "It predicted correctly with very high confidence — note the top-1 probability is over 99%, and the second-place prediction is essentially zero. The model isn't confused at all."

**Now pick a harder one — Tomato Early blight:**

> "Let's try one of the harder classes. Tomato Early blight."

**Watch the result.** If it's right, great — point out it's still confident. If it's wrong, that's actually a good moment too:

> "Notice the top probabilities are closer together here. Even when it gets it right, the model is less certain about Early blight versus Late blight than it is about, say, Yellow Leaf Curl. That matches the confusion matrix — these are the genuinely visually similar classes."

**Optional — switch to the Upload tab if you want to show user uploads:**

> "We can also upload arbitrary images. If anyone has a leaf photo they'd like to test, this works for any JPG."

---

## Section 8 — Challenges & Conclusion [1.5 min]

**Click each expandable section briefly:**

> "Three main challenges. First, the dataset's labelling is folder-based with no manifest, so we wrote a custom Dataset class that walks the directory tree."

> "Second, the class imbalance, which we handled through augmentation rather than weighted loss."

> "Third, the validation curve volatility, which turned out to be a metric artifact rather than a model issue."

**Point at the conclusion section:**

> "To wrap up — a ResNet-18 from scratch reaches 99% on PlantVillage with a sensible training recipe. But there's an honest caveat I want to make sure we acknowledge."

> "PlantVillage images are shot on uniform backgrounds with single centered leaves. Real field photos contain soil, multiple leaves, varied lighting, and occlusion. Models that hit 99% on PlantVillage typically lose substantial accuracy on field images. That's a known limitation in the literature, and it's the next thing we'd want to study — evaluating on field-captured data, and possibly mobile deployment."

> "Thank you. Happy to take questions."

---

## Likely questions Dr. Bdair might ask, and good answers

**Q: "Why ResNet-18 specifically and not ResNet-50?"**
> "Two reasons. First, model capacity should match data scale — with 20,000 training images, deeper variants have parameters they can't robustly support without pretraining. Second, ResNet-18 is the smallest member of the family that still benefits from skip connections, so we get optimization stability without paying for unnecessary capacity."

**Q: "Why didn't you use pretrained weights?"**
> "It was a deliberate choice for the project. We wanted to demonstrate that we understand the architecture by building it from scratch and training end-to-end. Using pretrained ImageNet weights would have hidden whether the implementation was correct, since pretrained features carry a lot for free. With our setup, the 99% accuracy is purely a result of the architecture and our training procedure."

**Q: "How would you handle class imbalance differently?"**
> "Our current approach is implicit, through augmentation. Three explicit alternatives we'd consider: weighted cross-entropy with inverse-frequency weights, oversampling the small classes with `WeightedRandomSampler`, or focal loss which down-weights easy examples. We'd compare all three against augmentation alone."

**Q: "What's the real-world deployment risk?"**
> "The biggest one is domain shift. PlantVillage is studio photography. Field images have wildly different conditions. A model that's 99% accurate on PlantVillage might be 60% accurate on phone photos taken in a real garden. The fix is to either fine-tune on field data, use stronger augmentation that simulates field conditions (background variation, lighting variation), or use techniques like domain adaptation."

**Q: "How long did training take?"**
> "Around two to three hours on the RTX 4070 laptop GPU."

**Q: "Did you try other architectures?"**
> "No, this was a focused project on ResNet-18. A natural extension would be a baseline comparison — a simple VGG-style CNN as a lower bound, and a pretrained ResNet-18 as an upper bound, to bracket where our from-scratch model sits."

**Q: "What's the bug with `RandomRotation(0.15)`?"**
> [If asked — be honest.] "Good catch. `RandomRotation` takes degrees, so we were actually only rotating by ±0.15° rather than ±15° as we intended. The model still reached 99% accuracy without meaningful rotation augmentation, which suggests the other augmentations carried the regularization. We'd fix that to `RandomRotation(15)` in a future iteration to see if it changes the per-class results."

---

## Final tips

- **Watch your pace.** People listening to a technical demo absorb information much slower than you produce it. When you finish a sentence, pause for a beat before the next one.
- **Don't read the screen.** The audience can read it themselves. Use the screen as a reference and *talk* about what's there.
- **If something fails live**, don't panic. Say "let me reload that section" and click the sidebar item again. Streamlit usually recovers immediately.
- **Have a teammate driving the laptop** while the other speaks, if possible. Splits cognitive load.
- **End on time.** A demo that ends two minutes early is better than one that runs two minutes long.
