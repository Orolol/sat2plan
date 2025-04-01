## The GAN is dead; long live the GAN!

## A Modern Baseline GAN

```
Yiwen Huang
Brown University
```
```
Aaron Gokaslan
Cornell University
```
```
Volodymyr Kuleshov
Cornell University
```
```
James Tompkin
Brown University
```
## Abstract

```
There is a widely-spread claim that GANs are difficult to train, and GAN archi-
tectures in the literature are littered with empirical tricks. We provide evidence
against this claim and build a modern GAN baseline in a more principled manner.
First, we derive a well-behaved regularized relativistic GAN loss that addresses
issues of mode dropping and non-convergence that were previously tackled via a
bag of ad-hoc tricks. We analyze our loss mathematically and prove that it admits
local convergence guarantees, unlike most existing relativistic losses. Second, this
loss allows us to discard all ad-hoc tricks and replace outdated backbones used
in common GANs with modern architectures. Using StyleGAN2 as an example,
we present a roadmap of simplification and modernization that results in a new
minimalist baseline—R3GAN (“Re-GAN”). Despite being simple, our approach
surpasses StyleGAN2 on FFHQ, ImageNet, CIFAR, and Stacked MNIST datasets,
and compares favorably against state-of-the-art GANs and diffusion models.
Code: https://www.github.com/brownvc/R3GAN
```
## 1 Introduction

```
Generative adversarial networks (GANs) let us generate high-quality images in a single forward pass.
However, the original objective in Goodfellowet al. [ 13 ], is notoriously difficult to optimize due to its
minimax nature. This leads to a fear that training might diverge at any point due to instability, and a
fear that generated images might lose diversity through mode collapse. While there has been progress
in GAN objectives [ 14 , 22 , 81 , 52 , 64 ], practically, the effects of brittle losses are still regularly felt.
This notoriety has had a lasting negative impact on GAN research.
A complementary issue—partly motivated by this instability—is that existing popular GAN backbones
like StyleGAN [ 29 , 31 , 30 , 32 ] use many poorly-understood empirical tricks with little theory. For
instance, StyleGAN uses a gradient penalized non-saturating loss [ 52 ] to increase stability (affecting
sample diversity), but then employs a minibatch standard deviation trick [ 28 ] to increase sample
diversity. Without tricks, the StyleGAN backbone still resembles DCGAN [ 60 ] from 2015, yet
it is still the common backbone of SOTA GANs such as GigaGAN [ 26 ] and StyleGAN-T [ 70 ].
Advances in GANs have been conservative compared to other generative models such as diffusion
models [ 20 , 78 , 33 , 34 ], where modern computer vision techniques such as multi-headed self
attention [ 87 ] and backbones such as preactivated ResNet [ 17 ], U-Net [ 63 ] and vision transformers
(ViTs) [ 9 ] are the norm. Given outdated backbones, it is not surprising that there is a widely-spread
belief that GANs do not scale in terms of quantitative metrics like Frechet Inception Distance [19].
```
```
We reconsider this situation: we show that by combining progress in objectives into a regularized
training loss, GANs gain improved training stability, which allows us to upgrade GANs with modern
backbones. First, we propose a novel objective that augments the relativistic pairing GAN loss
(RpGAN; [ 22 ]) with zero-centered gradient penalties [ 52 , 64 ], improving stability [ 14 , 64 , 52 ]. We
show mathematically that gradient-penalized RpGAN enjoys the same guarantee of local convergence
as regularized classic GANs, and that removing our regularization scheme induces non-convergence.
```
```
38th Conference on Neural Information Processing Systems (NeurIPS 2024).
```
# arXiv:2501.05441v1 [cs.LG] 9 Jan 2025


Once we have a well-behaved loss, none of the GAN tricks are necessary [ 28 , 31 ], and we are
free to engineer a modern SOTA backbone architecture. We strip StyleGAN of all its features,
identify those that are essential, then borrow new architecture designs from modern ConvNets
and transformers [ 48 , 97 ]. Briefly, we find that proper ResNet design [ 17 , 67 ], initialization [ 99 ],
and resampling [ 29 , 31 , 32 , 100 ] are important, along with grouped convolution [ 95 , 5 ] and no
normalization [ 31 , 34 , 14 , 88 , 4 ]. This leads to a design that is simpler than StyleGAN and improves
FID performance for the same network capacity (2.75 vs. 3.78 on FFHQ-256).

In summary, our work first argues mathematically that GANs need not be tricky to train via an
improved regularized loss. Then, it empirically develops a simple GAN baseline that, without any
tricks, compares favorably by FID to StyleGAN [ 29 , 31 , 32 ], other SOTA GANs [ 3 , 42 , 94 ], and
diffusion models [20, 78, 86] across FFHQ, ImageNet, CIFAR, and Stacked MNIST datasets.

## 2 Serving Two Masters: Stability and Diversity with RpGAN+R 1 +R 2

In defining a GAN objective, we tackle two challenges: stability and diversity. Some previous work
deals with stability [ 29 , 31 , 32 ] and other previous work deals with mode collapse [ 22 ]. To make
progress in both, we combine a stable method with a simple regularizer that is grounded by theory.

2.1 Traditional GAN

A traditional GAN [ 13 , 57 ] is formulated as a minimax game between a discriminator (or critic)Dψ
and a generatorGθ. Given real datax∼pDand fake datax∼pθproduced byGθ, the most general
form of a GAN is given by:

```
L(θ,ψ) =Ez∼pz[f(Dψ(Gθ(z)))] +Ex∼pD[f(−Dψ(x))] (1)
```
whereGtries to minimizeLwhileDtries to maximize it. The choice offis flexible [ 50 , 44 ]. In
particular,f(t) =−log(1 +e−t)recovers the classic GAN by Goodfellowet al. [ 13 ]. For the rest
of this work, this will be our choice off[57].

It has been shown that Equation 1 has convex properties whenpθcan be optimized directly [ 13 , 81 ].
However, in practical implementations, the empirical GAN loss typically shifts fake samples beyond
the decision boundary set byD, as opposed to directly updating the density functionpθ. This
deviation leads to a significantly more challenging problem, characterized by susceptibility to two
prevalent failure scenarios: mode collapse/dropping^1 and non-convergence.

2.2 Relativisticf-GAN

We employ a slightly different minimax game named relativistic pairing GAN (RpGAN) by Jolicoeur-
Martineauet al. [22] to address mode dropping. The general RpGAN is defined as:

```
L(θ,ψ) =Exz∼∼ppz
D
```
```
[f(Dψ(Gθ(z))−Dψ(x))] (2)
```
Although Eq. 2 differs only slightly from Eq. 1, evaluating this critic difference has a fundamental
impact on the landscape ofL. Since Eq. 1 merely requiresDto separate real and fake data, in the
scenario where all real and fake data can be separated by a single decision boundary, the empirical
GAN loss encouragesGto simply move all fake samples barely past this single boundary—this
degenerate solution is what we observe as mode collapse/dropping. Sunet al. [ 81 ] characterize such
degenerate solutions as bad local minima in the landscape ofL, and show that Eq. 1 hasexponentially
manybad local minima. The culprit is the existence of a single decision boundary that naturally
arises when real and fake data are considered in isolation. RpGAN introduces a simple solution by
coupling real and fake data,i.e. a fake sample is critiqued by its realnessrelative toa real sample,
which effectively maintains a decision boundary in the neighborhood ofeachreal sample and hence
forbids mode dropping. Sunet al. [ 81 ] show that the landscape of Eq. 2 contains no local minima
that correspond to mode dropping solutions, and that every basin is a global minimum.

(^1) While mode collapse and mode dropping are technically distinct issues, they are used interchangeably in
this context to describe the common problem wheresupp(pθ)does not comprehensively coversupp(pD).
Mode collapse refers to the generator producing a limited diversity of samples (i.e., one image for the entire
distribution), whereas mode dropping involves the generator failing to represent certain modes of the data
distribution (ignoring entire subsets of the training distribution).


2.3 Training Dynamics of RpGAN

Although the RpGAN landscape result [ 81 ] allows us to address mode dropping, the training dynamics
of RpGAN have yet to be studied. The ultimate goal of Eq. 2 is to find the equilibrium(θ∗,ψ∗)
such thatpθ∗=pDandDψ∗is constant everywhere onpD. Sunet al. [ 81 ] show thatθ∗is globally
reachable along a non-increasing trajectory in the landscape of Eq. 2 under reasonable assumptions.
However, the existence of such a trajectory does not necessarily mean that gradient descent will find it.
Jolicoeur-Martineauet al. show empirically that unregularized RpGAN does not perform well [22].

Proposition I.(Informal)Unregularized RpGAN does not always converge using gradient descent.

We confirm this proposition with a proof in Appendix B. We show analytically that RpGAN does
not converge for certain types ofpD, such as ones that approach a delta distribution. Thus, further
regularization is necessary to fill in the missing piece of a well-behaved loss.

Zero-centered gradient penalties. To tackle RpGAN non-convergence, we explore gradient
penalties as the solution since it is proven that zero-centered gradient penalties (0-GP) facilitate
convergent training for classic GANs [52]. The two most commonly-used 0-GPs areR 1 andR 2 :

```
R 1 (ψ) =
```
```
γ
2
```
```
Ex∼pD
```
```
h
∥∇xDψ∥^2
```
```
i
```
```
R 2 (θ,ψ) =
```
```
γ
2
```
```
Ex∼pθ
```
```
h
∥∇xDψ∥^2
```
```
i (3)
```
R 1 penalizes the gradient norm ofDon real data, andR 2 penalizes the gradient norm ofDon fake
data. Analysis on the training dynamics of GANs has thus far focused on local convergence [ 55 , 51 ,
52 ],i.e., whether the training at least converges when(θ,ψ)are in a neighborhood of(θ∗,ψ∗). In
such a scenario, the convergence behavior can be analyzed [ 55 , 51 , 52 ] by examining the spectrum
of the Jacobian of the gradient vector field(−∇θL,∇ψL)at(θ∗,ψ∗). The key insight here is that
whenGalready produces the true distribution, we want∇xD= 0, so thatGis not pushed away
from its optimal state, and thus the training does not oscillate.R 1 andR 2 impose such a constraint
whenpθ=pD. This also explains why earlier attempts at gradient penalties, such as the one-centered
gradient penalty (1-GP) in WGAN-GP [ 14 ], fail to achieve convergent training [ 52 ] as they still
encourageDto have a non-zero slope whenGhas reached optimality.

Since the same insight also applies to RpGAN, we extend our previous analysis and show that:

Proposition II.(Informal)RpGAN withR 1 orR 2 regularization is locally convergent subject to
similar assumptions as inMeschederet al. [52].

In Appendix C, our proof similarly analyzes the eigenvalues of the Jacobian of the regularized
RpGAN gradient vector field at(θ∗,ψ∗). We show that all eigenvalues have a negative real part; thus,
regularized RpGAN is convergent in a neighborhood of(θ∗,ψ∗)for small enough learning rates [ 52 ].

Discussion. Another line of work [ 64 ] linksR 1 andR 2 to instance noise [ 75 ] as its analytical
approximation. Roth et al. [ 64 ] showed that for the classic GAN [ 13 ] by Goodfellowet al.,R 1
approximates convolvingpDwith the density function ofN(0,γI), up to additional weighting and
a Laplacian error term.R 2 likewise approximates convolvingpθwithN(0,γI)up to similar error
terms. The Laplacian error terms fromR 1 ,R 2 cancel whenDψapproachesDψ∗. We do not extend
Rothet al.’s proof [ 64 ] to RpGAN; however, this approach might provide complimentary insights to
our work, which follows the strategy of Meschederet al. [52].

2.4 A Practical Demonstration

We experiment with how well-behaved our loss is on StackedMNIST [ 46 ] which consists of 1000
uniformly-distributed modes. The network is a small ResNet [ 17 ] forGandDwithout any normal-
ization layers [ 21 , 91 , 1 , 85 ]. Through the use of a pretrained MNIST classifier, we can explicitly
measure how many modes ofpDare recovered bypθ. Furthermore, we can estimate the reverse KL
divergence between the fake and real samplesDKL(pθ∥pD)via the KL divergence between the
categorical distribution ofpθand the true uniform distribution.


```
100
```
```
101
```
```
102
```
```
103
```
```
Generator loss
```
```
RpGAN + R 1 + R 2
GAN + R 1 + R 2
RpGAN + R 1
GAN + R 1
```
Figure 1: GeneratorGloss for different objec-
tives over training. Regardless of which objective
is used, training diverges with onlyR 1 and suc-
ceeded with bothR 1 andR 2. Convergence failure
with onlyR 1 was noted by Lee et al. [42].

```
Loss # modes↑ DKL↓
RpGAN+R 1 +R 2 1000 0. 0781
GAN+R 1 +R 2 693 0. 9270
RpGAN+R 1 Fail Fail
GAN+R 1 Fail Fail
```
```
Table 1: StackedMNIST [ 46 ] result for each loss
function. The maximum possible mode coverage
is 1000. “Fail” indicates that training diverged
early on.
```
A conventional GAN loss withR 1 , as used by Mescheder et al. [ 52 ] and the StyleGAN series [ 29 , 31 ,
32 ], diverges quickly (Fig. 1). Next, while theoretically sufficient for local convergence, RpGAN
with onlyR 1 regularization is also unstable and diverges quickly^2. In each case, the gradient ofDon
fake samples explodes when training diverges. With bothR 1 andR 2 , training becomes stable for
both the classic GAN and RpGAN. Now stable, we can see that the classic GAN suffers from mode
dropping, whereas RpGAN achieves full mode coverage (Tab. 1) and reducesDKLfrom 0.9270 to
0.0781. As a point of contrast, StyleGAN [ 29 , 31 , 30 , 32 ] uses the minibatch standard deviation trick
to reduce mode dropping, improving mode coverage from 857 to 881 on StackedMNIST^3 and with
barely any improvement onDKL[28].

R 1 alone is not sufficient for globally-convergent training. While a theoretical analysis of this is
difficult, our small demonstration still provides insights into the assumptions of our convergence proof.
In particular, the assumption that(θ,ψ)are sufficiently close to(θ∗,ψ∗)is highly unlikely early in
training. In this scenario, ifDis sufficiently powerful, regularizingDsolely on real data is not likely
to have much effect onD’s behavior on fake data and so training can fail due to an ill-behavedD
on fake data. This observation has been made by previous studies [ 84 , 83 ] specifically for empirical
GAN training, that regularizing an empirical discriminator with onlyR 1 leads to gradient explosion
on fake data due to the memorization of real samples.

Thus, the practical solution is to regularizeDon both real and fake data. The benefit of doing so can
be viewed from the insight of Rothet al. [ 64 ]: that applyingR 1 andR 2 in conjunction smooths both
pDandpθwhich makes learning easier than only smoothingpD. We also find empirically that with

bothR 1 andR 2 in place,Dtends to satisfyEx∼pD

```
h
∥∇xD∥^2
```
```
i
≈Ex∼pθ
```
```
h
∥∇xD∥^2
```
```
i
even early in
```
the training. Jolicoeur-Martineauet al. [ 23 ] show that in this caseDbecomes a maximum margin
classifier—but if only one regularization term is applied, this does not hold. Additionally, having
roughly the same gradient norm on real and fake data potentially reduces discriminator overfitting, as
Fanget al. [ 10 ] observe that the gradient norm on real and fake data diverges whenDstarts to overfit.

## 3 A Roadmap to a New Baseline — R3GAN

The well-behaved RpGAN +R 1 +R 2 loss alleviates GAN optimization problems, and lets us proceed
to build a minimalist baseline—R3GAN—with recent network backbone advances in mind [ 48 , 97 ].
Rather than simply state the new approach, we will draw out a roadmap from the StyleGAN
baseline [ 30 ]. This model (Config A; identical to [ 30 ]) consists of a VGG-like [ 73 ] backbone forG,
a ResNetD, a few techniques that facilitate style-based generation, and many tricks that serve as
patches to the weak backbone. Then, we remove all non-essential features of StyleGAN2 (Config B),
apply our loss function (Config C), and gradually modernize the network backbone (Config D-E).

(^2) Varyingγfrom 0.1 to 100 does not stabilize training.
(^3) These numbers are from Karraset al. [ 28 ], Table 4. "857" corresponds to a low-capacity version of a progressive
GAN and "881" adds the minibatch standard deviation trick. Further comparisons via loss curves are difficult
since progressive GAN is a substantially different model than the small ResNet we use for this experiment.


We evaluate each configuration on FFHQ 256 × 256 [ 29 ]. Network capacity is kept roughly the same
for all configurations—bothGandDhave about 25 M trainable parameters. Each configuration is
trained untilDsees 5 M real images. We inherit training hyperparameters (e.g., optimizer settings,
batch size, EMA decay length) from Config A unless otherwise specified. We tune the training
hyperparameters for our final model and show the converged result in Sec. 4.

```
Configuration FID↓ G #params D #params
A StyleGAN2 7.516 24.767M 24.001M
B Stripped StyleGAN
```
- znormalization
- Minibatch stddev
- Equalized learning rate
- Mapping network
- Style injection
- Weight mod / demod
- Noise injection
- Mixing regularization
- Path length regularization
- Lazy regularization

```
12.46 18.890M 23.996M
```
```
C Well-behaved Loss
+ RpGAN loss 11.77 18.890M 23.996M
+R 2 gradient penalty 11.
D ConvNeXt-ify pt. 1
+ ResNet-ify G&D 10.17 23.400M 23.282M
```
- Output skips 9.950 23.378M
E ConvNeXt-ify pt. 2
+ ResNeXt-ify G&D 7.507 23.188M 23.091M
+ Inverted bottleneck 7.045 23.058M 23.010M
Table 2: Effect of our simplification and modernization
efforts evaluted on FFHQ-256.

Minimum baseline (Config B). We
strip away all StyleGAN2 features, re-
taining only the raw network backbone
and basic image generation capability.
The features fall into three categories:

- Style-based generation: mapping net-
    work [ 29 ], style injection [ 29 ], weight
    modulation/demodulation [ 31 ], noise
    injection [29].
- Image manipulation enhancements:
    mixing regularization [ 29 ], path
    length regularization [31].
- Tricks: znormalization [ 28 ], mini-
    batch stddev [ 28 ], equalized learning
    rate [28], lazy regularization [31].

Following [ 69 , 70 ], we reduce the dimen-
sion ofzto 64. The absence of equal-
ized learning rate necessitates a lower
learning rate, reduced from 2.5× 10 -3to
5 × 10 -5. Despite a higher FID of 12.
than Config-A, this simplified baseline
produces reasonable sample quality and stable training. We compare this with DCGAN [ 60 ], an early
attempt at image generation. Key differences include:

```
a) Convergent training objective withR 1 regularization.
b) Smaller learning rate, avoiding momentum optimizer (Adamβ 1 = 0).
c) No normalization layer inGorD.
d) Proper resampling via bilinear interpolation instead of strided (transposed) convolution.
e) Leaky ReLU in bothGandD, no tanh in the output layer ofG.
f) 4×4 constant input forG, output skips forG, ResNetD.
```
Experimental findings from StyleGAN.Violating a), b), or c) often leads to training failures. Gidelet
al. [ 11 ] show thatnegativemomentum can improve GAN training dynamics. Since optimal negative
momentum is another challenging hyperparameter, we do not use any momentum to avoid worsening
GAN training dynamics. Studies suggest normalization layers harm generative models [ 31 , 34 ].
Batch normalization [ 21 ] often cripples training due to dependencies across multiple samples, and is
incompatible withR 1 ,R 2 , or RpGAN that assume independent handling of each sample. Weaker
data-independent normalizations [ 31 , 34 ] might help; we leave this for future work. Early GANs may
succeed despite violating a) and c), possibly constituting a full-rank solution [52] to Eq. 1.

Violations of d) or e) do not significantly impair training stability but negatively affect sample
quality. Improper transposed convolution can cause checkerboard artifacts, unresolved even with
subpixel convolution [ 72 ] or carefully tuned transposed convolution unless a low-pass filter is applied.
Interpolation methods avoid this issue, varying from nearest neighbor [ 28 ] to Kaiser filters [ 32 ]. We
use bilinear interpolation for simplicity. For activation functions, smooth approximations of (leaky)
ReLU, such as Swish [ 61 ], GELU [ 18 ], and SMU [ 2 ], worsen FID. PReLU [ 15 ] marginally improves
FID but increases VRAM usage, so we use leaky ReLU.

All subsequent configurations adhere to a) through e). Violation of f) is acceptable as it pertains to
the network backbone of StyleGAN2 [31], modernized in Config D and E.

Well-behaved loss function (Config C). We use the loss function proposed in Section 2 and this
reduces FID to 11.65. We hypothesize that the network backbone in Config B is the limiting factor.

General network modernization (Config D). First, we apply the 1-3-1 bottleneck ResNet archi-
tecture [ 16 , 17 ] to bothGandD. This is the direct ancestor of all modern vision backbones [ 48 , 97 ].


```
(a) Overall view (b) StyleGAN2 architecture blocks [31] (Config A) (c) Ours (Config E)
Figure 2:Architecture comparison.For image generation,GandDare often both deep ConvNets
with either partially or fully symmetric architectures.(a)StyleGAN2 [31]Guses a network to map
noise vectorzto an intermediate style spaceW. We use a traditional generator as style mapping is
not necessary for a minimal working model.(b)StyleGAN2’s building blocks have intricate layers
but are themselves simple, with a ConvNet architecture from 2015 [ 38 , 73 , 16 ]. ResNet’s identity
mapping principle is also violated in the discriminator.(c)We remove tricks and modernize the
architecture. Our design has clean layers with a more powerful ConvNet architecture.
```
```
We also incorporate principles discovered in Config B and various modernization efforts from
ConvNeXt [48]. We categorize the roadmap of ConvNeXt as follows:
```
```
i.Consistently beneficial: i.1) increased width with depthwise convolution, i.2) inverted bottleneck,
i.3) fewer activation functions, and i.4) separate resampling layers.
ii.Negligible performance gain: ii.1) large kernel depthwise conv. with fewer channels, ii.2) swap
ReLU with GELU, ii.3) fewer normalization layers, and ii.4) swap batch norm. with layer norm.
```
iii.Irrelevant to our setting: iii.1) improved training recipe, iii.2) stage ratio, and iii.3) ‘patchify’ stem.

```
We aim to apply i) to our model, specifically i.3 and i.4 for the classic ResNet, while reserving i.1 and
i.2 for Config E. Many aspects of ii) were introduced merely to mimic vision transformers [ 47 , 9 ]
without yielding significant improvements [ 48 ]. ii.3 and ii.4 are inapplicable due to our avoidance
of normalization layers following principle c). ii.2 contradicts our finding that GELU deteriorates
GAN performance, thus we use leaky ReLU per principle e). Liuet al. emphasize large conv. kernels
(ii.1) [ 48 ], but this results in slightly worse performance compared to wider 3×3 conv. layers, so we
do not adopt this ConvNeXt design choice.
```
```
Neural network architecture details. Given i.3, i.4, and principles c), d), and e), we can replace the
StyleGAN2 backbone with a modernized ResNet. We use a fully symmetric design forGandDwith
25 M parameters each, comparable to Config-A. The architecture is minimalist: each resolution stage
has one transition layer and two residual blocks. The transition layer consists of bilinear resampling
and an optional 1×1 conv. for changing spatial size and feature map channels. The residual block
includes five operations: Conv1× 1 →Leaky ReLU→Conv3× 3 →Leaky ReLU→Conv1×1, with
the final Conv1×1 having no bias term. For the 4×4 resolution stage, the transition layer is replaced
by a basis layer forGand a classifier head forD. The basis layer, similar to StyleGAN [ 29 , 31 ],
uses 4×4 learnable feature maps modulated byzvia a linear layer. The classifier head uses a global
4 ×4 depthwise conv. to remove spatial extent, followed by a linear layer to produceD’s output. We
maintain the width ratio for each resolution stage as in Config A, making the stem width 3×as wide
due to the efficient 1×1 conv. The 3×3 conv. in the residual block has a compression ratio of 4,
following [16, 17], making the bottleneck width 0.75×as wide as Config A.
To avoid variance explosion due to the lack of normalization, we employ fix-up initialization [ 99 ]: We
zero-initialize the last convolutional layer in each residual block and scale down the initialization of
the other two convolutional layers in the block byL−^0.^25 , whereLis the number of residual blocks.
We avoid other fix-up tricks, such as excessive bias terms and a learnable multiplier.
```

Bottleneck modernization (Config E). Now that we have settled on the overall architecture, we
investigate how the residual block can be modernized, specifically i.1) and i.2). First, we explore
i.1 and replace the 3×3 convolution in the residual block with a grouped convolution. We set the
group size to 16 rather than 1 (i.e. depthwise convolution as in ConvNeXt) as depthwise convolution
is highly inefficient on GPUs and is not much faster than using a larger group size. With grouped
convolution, we can reduce the bottleneck compression ratio to two given the same model size. This
increases the width of the bottleneck to 1.5×as wide as Config A. Finally, we notice that the compute
cost of grouped convolution is negligible compared to 1×1 convolution, and so we seek to enhance
the capacity of grouped convolution. We apply i.2), which inverts the bottleneck width and the stem
width, and which doubles the width of grouped convolutions without any increase in model size.
Figure 2 depicts our final design, which reflects modern CNN architectures.

## 4 Experiments Details

4.1 Roadmap Insights on FFHQ-256 [29]

As per Table 2, Config A (vanilla StyleGAN2) achieves an FID of 7.52 using the official implementa-
tion on FFHQ-256. Config B with all tricks removed achieves an FID of 12.46—performance drops
as expected. Config C, with a well-behaved loss, achieves an FID of 11.65. But, now training is
sufficiently stable to improve the architecture.

Config D, which improvesGandDbased on the classic ResNet and ConvNeXt findings, achieves
an FID of 9.95. The output skips of the StyleGAN2 generator are no longer useful given our new
architecture; including them produces a worse FID of 10.17. Karraset al. find that the benefit of
output skips is mostly related to gradient magnitude dynamics [ 32 ], and this has been addressed by
our ResNet architecture. For StyleGAN2, Karraset al. conclude that a ResNet architecture is harmful
toG[ 31 ], but this is not true in our case as their ResNet implementation is considerably different
from ours: 1) Karraset al. use one 3-3 residual block for each resolution stage, while we have a
separate transition layer and two 1-3-1 residual blocks; 2) i.3) and i.4) are violated as they do not have
a linear residual block [ 67 ] and the transition layer is placed on the skip branch of the residual block
rather than the stem; 3) the essential principle of ResNet [ 16 ]—identity mapping [ 17 ]—is violated
as Karraset al. divide the output of the residual block by

### √

2 to avoid variance explosion due to the
absence of a proper initialization scheme.

For Config E, we conduct two experiments that ablate i.1 (increased width with depthwise conv.)
and i.2 (an inverted bottleneck). We add GroupedConv and reduce the bottleneck compression
ratio to two given the same model size. Each bottleneck is now 1.5×the width of Config A, and
the FID drops to 7.51, surpassing the performance of StyleGAN2. By inverting the stem and the
bottleneck dimensions to enhance the capacity of GroupedConv, our final model achieves an FID of
7.05, exceeding StyleGAN2.

4.2 Mode Recovery — StackedMNIST [53]

```
Model # modes↑ DKL↓
DCGAN [60] 99 3.
VEEGAN [80] 150 2.
WGAN-GP [14] 959 0.
PacGAN [46] 992 0.
StyleGAN2 [31] 940 0.
PresGAN [8] 1000 0.
Adv. DSM [24] 1000 1.
VAEBM [93] 1000 0.
DDGAN [94] 1000 0.
MEG [39] 1000 0.
Ours—Config E 1000 0.
```
```
Table 3: StackedMNIST 1000-mode coverage.
```
We repeat the earlier experiment in 1000-mode con-
vergence on StackedMNIST (unconditional genera-
tion), but this time with our updated architecture and
with comparisons to SOTA GANs and likelihood-
based methods (Tab. 3, Fig. 5). One advantage
brought up of likelihood-based models such as dif-
fusion over GANs is that they achieve mode cover-
age [ 7 ]. We find that most GANs struggle to find
all modes. But, PresGAN [ 8 ], DDGAN [ 94 ], and
our approach are successful. Further, our method
outperforms all other tested GAN models in term
of KL divergence.

4.3 FID — FFHQ-256 [29] (Optimized)

We train Config E model until convergence and with optimized hyperparameters and training schedule
on FFHQ at 256×256 (unconditional generation) (Tab. 4, Figs. 4 and 6). Please see our supplemental
material for training details. Our model outperforms existing StyleGAN methods, plus four more


```
Model NFE↓ FID↓
StyleGAN2 [31] 1 3.
StyleGAN3-T [32] 1 4.
StyleGAN3-R [32] 1 3.
LDM [62] 200 4.
ADM (DDIM) [7, 49] 500 8.
ADM (DPM-Solver) [7, 49] 500 8.
Diffusion Autoencoder [59, 49] 500 5.
Ours—Config E 1 2.
With ImageNet feature leakage [41]:
PolyINR* [74] 1 2.
StyleGAN-XL* [69] 1 2.
StyleSAN-XL* [82] 1 1.
```
```
Table 4: FFHQ-256. * denotes models that leak
ImageNet features.
```
```
Model NFE↓ FID↓
StyleGAN2 [31, 45] 1 3.
MSG-GAN [27, 45] 1 2.
Anycost GAN [45] 1 2.
VE [78, 33] 79 25.
VP [78, 33] 79 3.
EDM [33] 79 2.
Ours—Config E 1 1.
```
```
Table 5: FFHQ-64.
```
recent diffusion-based methods. On this common dataset experimental setting, many methods (not
listed here) use the bCR [ 101 ] trick—this has only been shown to improve performance on FFHQ-
(not even at different resolutions of FFHQ) [101, 98]. We do not use this trick.

### 4.4 FID — FFHQ-64 [33]

To compare with EDM [ 33 ] directly, we evaluate our model on FFHQ at 64×64 resolution. For this,
we remove the two highest resolution stages of our 256×256 model, resulting in a generator that is
less than half the number of parameters as EDM. Despite this, our model outperforms EDM on this
dataset and needs one function evaluation only (Tab. 5).

4.5 FID — CIFAR-10 [37] Model NFE↓ FID↓
BigGAN [3] 1 14.
TransGAN [87] 1 9.
ViTGAN [42] 1 6.
DDGAN [94] 4 3.
Diffusion StyleGAN2 [90] 1 3.
StyleGAN2 + ADA [30] 1 2.
StyleGAN3-R + ADA [32, 25] 1 10.
DDPM [20] 1000 3.
DDIM [76] 50 4.
VE [78, 33] 35 3.
VP [78, 33] 35 2.
Ours—Config E 1 1.
With ImageNet feature leakage [41]:
StyleGAN-XL* [69] 1 1.

```
Table 6: CIFAR-10 performance.
```
We train Config E model until convergence and
with optimized hyperparameters and training sched-
ule on CIFAR-10 (conditional generation) (Tab. 6,
Fig. 8). Our method outperforms many other GANs
by FID even though the model has relatively small
capacity. For instance, StyleGAN-XL [ 69 ] has 18
M parameters in the generator and 125 M parame-
ters in the discriminator, while our model has a 40
M parameters between the generator and discrim-
inator combined (Fig. 3). Compared to diffusion
models like LDM or ADM, GAN inference is sig-
nificantly cheaper as it requires only one network
function evaluation compared to the tens or hun-
dreds of network function evaluations for diffusion
models without distillation.

```
Figure 3: Millions of parameters vs. FID-50K
(log scale) on CIFAR-10. Lower is better.
```
Many state-of-the-art GANs are derived from Pro-
jected GAN [ 68 ], including StyleGAN-XL [ 69 ] and
the concurrent work of StyleSAN-XL [ 82 ]. These
methods use a pre-trained ImageNet classifier in
the discriminator. Prior work has shown that a pre-
trained ImageNet discriminator can leak ImageNet
features into the model [ 41 ], causing the model to
perform better when evaluating on FID since it re-
lies on a pre-trained ImageNet classifier for the loss.
But, this does not improve results in perceptual stud-
ies [ 41 ]. Our model produces its low FID without
any ImageNet pre-training.


```
Model NFE↓ FID↓
DDPM++ [35] 1000 8.
VDM [36] 1000 7.
MSGAN [27, 56] 1 12.
ADM [7] 1000 3.
DDPM-IP [56] 1000 2.
Ours—Config E 1 1.
With ImageNet feature leakage [41]:
StyleGAN-XL* [69] 1 1.
```
```
Table 7: ImageNet-32.
```
```
Model NFE↓ FID↓
BigGAN-deep [3] 1 4.
DDPM [20] 250 11.
DDIM [76] 50 13.
ADM [7] § 250 2.
EDM [33] 79 2.
CT [79] 2 11.
CD [79] 3 4.
iCT-deep [77] 2 2.
DMD [96] 1 2.
Ours—Config E 1 2.
With ImageNet feature leakage [41]:
StyleGAN-XL* [69] 1 1.
```
```
Table 8: ImageNet-64.§:deterministic sampling.
```
4.6 FID — ImageNet-32 [6]

We train Config E model until convergence and with optimized hyperparameters and training schedule
on ImageNet-32 (conditional generation). We compare against recent GAN models and recent
diffusion models in Table 7. We adjust the number of parameters in the generator of our model
to match StyleGAN-XL [ 69 ]’s generator (84M parameters). Specifically, we make the model
significantly wider to match. Our method achieves comparable FID despite using a 60% smaller
discriminator (Tab. 7) and despite not using a pre-trained ImageNet classifier.

4.7 FID — ImageNet-64 [6]

We evaluate our model on ImageNet-64 to test its scalability. We stack another resolution stage on
our ImageNet-32 model, resulting in a generator of 104 M parameters. This model is nearly 3×
smaller than diffusion-like models [ 7 , 33 , 79 , 77 ] that rely on the ADM backbone, which contains
about 300 M parameters. Despite the smaller model size and that our model generates samples in one
step, it outperforms larger diffusion models with many NFEs on FID (Tab. 8).

4.8 Recall

We evaluate the recall [ 40 ] of our model on each dataset to quantify sample diversity. In general, our
model achieves a recall that is similar to or marginally worse than the diffusion model counterpart,
yet superior to existing GAN models. For CIFAR-10, the recall of our model peaked at 0.57; as a
point of comparison, StyleGAN-XL [ 69 ] has a worse recall of 0.47 despite its lower FID. For FFHQ,
we obtain a recall of 0.53 at 64×64 and 0.49 at 256×256, whereas StyleGAN2 [ 31 ] achieved a recall
of 0.43 on FFHQ-256. Our ImageNet-32 model achieved a recall of 0.63; comparable to ADM [ 7 ].
Our ImageNet-64 model achieved recall 0.59. While this is slightly worse than≈0.63 that many
diffusion models achieve, it is better than BigGAN-deep [3] which achieved a recall of 0.48.

## 5 Discussion and Limitations

We have shown that a simplification of GANs is possible for image generation tasks, built upon a more
stable RpGAN+R 1 +R 2 objective with mathematically-demonstrated convergence properties that
still provides diverse output. This stability is what lets us re-engineer a modern network architecture
without the tricks of previous methods, producing the R3GAN model with competitive FID on the
common datasets of Stacked-MNIST, FFHQ, CIFAR-10, and ImageNet as an empirical demonstration
of the mathematical benefits.

The focus of our work is to elucidate the essential components of a minimum GAN for image
generation. As such, we prioritize simplicity over functionality—we do not claim to beat the
performance of every existing model on every dataset or task; merely to provide a new simple


```
Figure 4: Qualitative examples of sample generation from our Config E on FFHQ-256.
```
baseline that converges easily. While this makes our model a possible backbone for future GANs,
it also means that it is not suitable to apply our model directly to downstream applications such
as image editing or controllable generation, as our model lacks dedicated features for easy image
inversion or disentangled image synthesis. For instance, we remove style injection functionality from
StyleGAN even though this has a clear use. We also omitted common techniques that have been
shown in previous literature to improve FID considerably. Examples include some form of adaptive
normalization modulated by the latent code [ 7 , 33 , 29 , 98 , 58 , 89 , 66 ], and using multiheaded self
attention at lower resolution stages [ 7 , 33 , 34 ]. We aim to explore these techniques in a future study.

Further, our work is limited in its evaluation of the scalability of R3GAN models. While they show
promising results on 64×64 ImageNet, we are yet to verify the scalability on higher resolution
ImageNet data or large-scale text to image generation tasks [12].

Finally, as a method that can improve the quality of generative models, it would be amiss not to men-
tion that generative models—especially of people—can cause direct harm (e.g., through personalized
deep fakes) and societal harm through the spread of disinformation (e.g., fake influencers).

## 6 Conclusion

This work introduced R3GAN, a new baseline GAN that features increased stability, leverages modern
architectures, and does not require ad-hoc tricks that are commonplace in existing GAN models.
Central to our approach is a regularized relativistic loss that provably features local convergence
and that improves the stability of GAN training. This stable loss enables us to ablate various tricks
that were previously necessary in GANs, and incorporate in their place modern deep architectures.
The resulting streamlined baseline achieves competitive performance to SOTA models within its
parameter size class. We anticipate that our backbone will help to drive future GAN research.


Acknowledgements. The authors thank Xinjie Jayden Yi for contributing to the proof and Yu
Cheng for helpful discussion. For compute, the authors thank Databricks Mosaic Research. Yiwen
Huang was supported by a Brown University Division of Research Seed Award, and James Tompkin
was supported by NSF CAREER 2144956. Volodymyr Kuleshov was supported by NSF CAREER
2145577 and NIH MIRA 1R35GM15124301.

