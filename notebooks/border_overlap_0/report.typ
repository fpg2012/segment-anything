#set text(font: ("New Computer Modern", "Noto Serif CJK SC"))

= 结果


\

算了100张图，数据集还是COCO MVal。

ps. 把所有Attention全部存下来不太现实，存4张图的所有global attention就要吃掉48GB的空间了。所以最后还是在线计算的，也不会很慢。

就结果来看，边缘patch受到的关注度可能略低于一半，而且层越靠后越低。

=== 边缘patch的判断

边缘提取目前用的是Canny算子，只要一个patch里面有包含边缘，就认为是边缘patch。所以一张图有可能大部分patch都是边缘patch。手调了一下Canny算子的参数，并且喂给Canny之前先简单高斯模糊了一下，对大部分图来说边缘提取结果还行。

=== Overlap Size

attention中的每行的top-100与边缘patch做交集，取交集的大小，除以100。

对于同一层的同一个head，在每张图片、attention每行取平均

#figure(
  grid(
    columns: (1fr, 1fr),
    image("./overlap-attn-block-7.svg"), image("./overlap-attn-block-15.svg"),
    image("./overlap-attn-block-23.svg"), image("./overlap-attn-block-31.svg")
  ),
  caption: "overlap size"
)

#pagebreak()

== Overlap Attention Sum

attention中的每行的top-100与边缘patch做交集，算交集部分的attention和。

对于同一层的同一个head，在每张图片、attention每行取平均

#figure(
  grid(
    columns: (1fr, 1fr),
    image("./overlap_attenntion_sum-attn-block-7.svg"), image("./overlap_attenntion_sum-attn-block-15.svg"),
    image("./overlap_attenntion_sum-attn-block-23.svg"), image("./overlap_attenntion_sum-attn-block-31.svg")
  ),
  caption: "overlap attention sum"
)

#pagebreak()

== Border Attention Sum

attention中的每行，算边缘patch的attention和

对于同一层的同一个head，在每张图片、attention每行取平均

#figure(
  grid(
    columns: (1fr, 1fr),
    image("./border_attention_sum-attn-block-7.svg"), image("./border_attention_sum-attn-block-15.svg"),
    image("./border_attention_sum-attn-block-23.svg"), image("./border_attention_sum-attn-block-31.svg")
  ),
  caption: "overlap attention sum"
)