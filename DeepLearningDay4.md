# 深層学習Day4
## 強化学習
### ・講義のまとめ

教師なし学習、教師あり学習、強化学習という学習分類の1つ．  

長期的に報酬を最大化できるように環境の中で行動できるエージェントを作ることを目標とする機械学習の1つの手法  
何かしら目的に向かって、自力で学習を進める学習方法．（ex. 人間が職場でする仕事）  
最初は未知の行動をとって模索する必要があるが、ある程度経つと過去の経験も活かすことも必要になる．“未知の探索”と“経験の利用”の2つを行き来しながらより良い状態を目指しながら学習を進めていくイメージ．  

![image](https://user-images.githubusercontent.com/87635559/126138175-c65f8403-d759-459d-bea1-a4d69173a75e.png)  

強化学習　“状態”に応じて優れた”方策”(行動)を見つけることが目標  

方策関数　π（ｓ）　エージェントがどんな行動をするかを決める関数、ある状態で価値関数が最大になるように行動を決める、（VやQを下に）どのような行動をとるべきかの確率を与える．  

価値関数　目標設定に当たる　大きく2種類ある  
　状態価値関数　V（ｓ）　環境の状態が価値を決める  
　行動価値関数　Q（ｓ, a）状態と行動、2つをもとに価値を決める  

Q学習  
・行動価値関数を行動するたびに更新することにより学習を進める  

関数近似法  
・価値関数や方策関数を関数近似する手法　→　関数なのでニューラルネットワークで近似できる  

上記2つを組み合わせることで理論的な発展を遂げた  

方策勾配法  
方策反復法　方策をモデル化して最適化する手法  
Θ(t+1) = Θ(t) + ε∇J(Θ)  
JはNNでの誤差関数に相当、ここでは期待収益を表す  
NNでは誤差関数を小さくに、強化学習では期待収益を大きくしたい．  

∇ΘJ(Θ) ＝ ∇ΘΣπΘ(a|s)Q(s,a)  
ある行動をとったときの価値をすべての行動パターンに対して足し合わせる  

### ・考察(および感想)
概念的にはある程度理解できたが、数学的な部分が浅くしか解説されていなかったので，おそらく結構難しいのだろう．  

### ・追加調査
応用事例としてマリオを強化学習で攻略する事例を見つけた．  

https://zakopilo.hatenablog.jp/entry/2021/01/30/214806  
https://qiita.com/karaage0703/items/e237887894f0f1382d58  

pytorch用のチュートリアルが用意されているようなので，E資格取得後にでも遊んでみたい．  

## AlphaGo
### ・講義のまとめ
強化学習が注目を集めたきっかけ  

AlphaGo Lee  
Value net(価値関数) … 畳み込みニューラルネットワーク 19x19(碁のマス目)x49(石、着手履歴、取れる石の数、手番、etc.)を入力，現局面の勝率を-1～1で出力(hyperbolic tangentで-∞～∞を-1～1に変換)  

Policy net(方策関数) … 畳み込みニューラルネットワーク　19x19(碁のマス目)x48 (石、着手履歴、取れる石の数、etc.)を入力，19x19マスの着手予想を出力  

RollOutPolicy … NNではなく線形の方策関数、高速に着手手段を出すために使用される．計算量が多いニューラルネットワークの代替として使われる．Policy netの1000倍速い．  
(高精度な手法の使用が都合が悪い場合に備えて、多少精度が落ちても高速に動くものを用意することは機械学習の常套手段)  

１．教師あり学習でRollOutPolicy, Policy netの学習をある程度進めておく  
２．強化学習によるPolicy netの学習  
３．強化学習によるValue netの学習  

強化学習の演算にはモンテカルロ木探索を用いる  

AlphaGo Zero  
AlphaGo Leeとの違い  
１．	教師あり学習を一切行わず強化学習のみ  
２．	特徴として入力するデータからヒューリスティックな要素を排除し、医師の配置のみ入力  
３．	Policy netとValue netを１のネットワークに統合  
４．	Residual Netを導入  
５．	モンテカルロ木探索からRollOutシミュレーションをなくした  

ネットワークの途中で枝割れして方策関数と価値関数ように２つの出力を出す  
ResidualBlockはResidualNetoworkが３９個連なっている  
![image](https://user-images.githubusercontent.com/87635559/126138961-c5ef59b1-6d6d-48c2-a2fc-3df7e35ea038.png)  

ResidualNetwork  
ショートカットをつくることで勾配消失、爆発を防ぐ，１００層を超えるネットワークでも学習が安定化  
基本構造は Convolution -> BatchNorm -> ReLu -> Convolution -> BatchNorm -> Add -> ReLu  
画像用の深いCNNでも使われる手法  
副作用：ショートカットによるスキップがある分、違うネットワークを仮想的に作っていることなるので、(疑似的に）色々なネットワークを通ってきたかのような出力が得られる  

![image](https://user-images.githubusercontent.com/87635559/126139087-c597b44a-d09f-404a-9998-20684bf00ca4.png)  

ResidualNetworkの工夫  
　Bottleneck … 1x1カーネルのconvolutionを利用し１層目で次元削減を行って３層目で次元を復元する３層構造  
　PreActivation … 並びをBatchNorm->Relu->Convolution^>BatchNorm->ReLu->Covolution->Addに変更  
　WideResNet … convolutionのフィルタ数をk倍にしたResNet  
　PyramidNet … WideResNetで幅が広がった直後の層に過度に負担がかかり制度を落とす原因になっていることを回避するために、段階的にではなく、各層でフィルタ数を増やしていくResNet  

### ・考察(および感想)
ResidualNetは要追加学習  

また講師の方が言われていた、CNN、RNN、プーリング、attention、活性化関数が基本パーツであり、この組み合わせにネットワークの名前がついているだけなので、ビビらないようにとのコメントは本質的をついていると思う．
AIは本当に沢山の論文，研究が発表されているが，あまり振り回されないようにしたい．


### ・追加調査
AlphaZero  
囲碁、チェス、将棋のいずれにも対応できるように汎用化させたAlphaGoの最新版  
https://ja.wikipedia.org/wiki/AlphaGo#AlphaZero  
https://wired.jp/2017/12/08/deepmind-alphazero/  


## 軽量化・高速化技術
### ・講義のまとめ
実用化に向けて準備、研究が進んでいる分野  

スマートフォンで動かしたい … 量子化、蒸留、プルーニング  

分散深層学習  
モデル、データ量は毎年１０倍ずつ計算量が増えている  
これに対応するため複数の計算資源を使って分散，並列的に計算を進める  
・データ並列化 … データを分割し複数のマシンで同じモデルを学習  
　同期型 … 各ワーカが計算を終えるのを待ち，パラメータ更新のタイミングをそろえ、モデル更新後に各ワーカで同じモデルを用いて次の計算を行う  
　非同期型 … 各ワーカはお互いの計算終了を待たない．各マシンの計算終了後にパラメータサーバにパラメータを送る．新たに学習を始めるときはパラメータサーバから最新のパラメータを参照する  
スピードは非同期型の方が速い  
非同期型は学習が不安定 → 同期型の方が高精度  

・モデル並列化 … モデルを各ワーカに分割してそれぞれのモデルで学習を行う．  
全てのデータで学習が終わった後で１つのモデルに復元．枝分かれ部分で分割することが主流．  
1台のPCで(GPUを複数つなぎ)学習を行うことが多い  
大きなモデル、パラメータ数が多いモデルほどスピードアップする，逆に小さいモデルだとかえって非効率になる場合もある．  


・GPU  
CPU … 高性能なコアが少数、複雑で連続的な処理が得意．イメージ：少数精鋭主義  
GPU …比較的低性能なコアが多数、簡単な並列処理が得意、ニューラルネットの学習は単純な行列演算が多いので向いている(元々ゲームのグラフィックス計算に使われていた)．イメージ：一般労働者  

GPU開発環境  
CUDA … Nvidiaの規格　現在の主流はこちら  
OpenCL … Intel, AMD, ARMなどで使われているオープンな規格  

量子化(Quantization)  
重みなどパラメータに関して64bit floatから32bitなどに精度を落とすことでメモリと演算処理の削減を行う  

64bit=8byte  
8byte x 1024 = 8KB  

32bit = 4byte  
2byte x 1024 = 2KB  

上記の例でも単純にメモリは半分になるし、計算も高速化する  

ただ、bitを落とすと数値計算の精度が落ちるので、やみくもに量子化すればよいというものではない  

メリット  
　計算速度  
　省メモリ  
デメリット  
　精度低下  

講師の方の結論としては多くの深層学習では16bitの精度で十分なので、GPUでの処理速度も高い16bitでよいとのこと  

蒸留  
精度の高いモデルから知識を継承させて軽量なモデルを作ること  
教師モデル　予測精度の高い複雑なモデル  
生徒モデル　教師モデルをもとに作られた軽量なモデル  
教師モデルの重みを固定し、生徒モデルの重みを更新していく  
誤差は教師モデルと生徒モデルのそれぞれの誤差を足し合わせて重みを更新していく  

プルーニング  
大量のパラメータから必要でないパラメータを削除すること  
重みが０に近い(閾値以下の)ニューロンを削除する  

かなり多くのパラメータを削除することができ，それほど精度も変わらない  
講義では94%のパラメータを削除しても精度は1%しか変わらない例が紹介されていた  

### ・考察(および感想)
これまで学習したことに比べて，かなりエンジニアリングな話題だった．
当然だが、実用化に当たっては数学やアルゴリズムによる工夫だけでなく、ハードウェアも含めて工夫、検討しなければならないということだろう

講師の方の得意分野なのか、生き生きしていたのが印象的だった

色々な手法について説明してくれたが，効果を比較するためにもぜひ演習サンプルコードを動かしてみたい内容だった  

## 応用モデル
### ・講義のまとめ
MobileNets … 画像向けのディープラーニングモデルの軽量化・高速化・高精度化(モバイルなネットワーク)　畳み込みの計算を工夫している  
　Depthwise Separable Convolutionという手法で以下の２つの計算に分割  
　Depthwise Convolution … カーネルのフィルタ数を１つに固定　(空間方向の計算)  
　Pointwise Convolution … フィルタ数は減らさず，カーネルのサイズを1x1に固定　(チャンネル方向の計算)  

![image](https://user-images.githubusercontent.com/87635559/126140857-28b76293-ff19-4750-8610-510527e1040c.png)  

DenseNet … 画像向けの有名CNN　Denseブロックを導入  
　Denseブロック … 前のブロックで処理された内容をチャンネルとして追加したものを入力とする  
　Tansition Layer … Denseブロックからの出力に対してダウンサンプリングを行い，チャンネルサイズを減らし、次のDenseブロックに渡す  
　
　ResNetとの違い  
　ResNetは前一層の入力のみ後方に渡る  
　DenseNetでは前方の各層から後方へ渡る  

　GrothRate（チャンネル数の成長率）をどのように調整するかが問題  
 
BatchNorm … レイヤー間を流れるデータをミニバッチ単位で正規化(複数画像のチャンネルをまとめて正規化)
　バッチサイズに影響を受けることが問題  
　講師曰く実際にはあまり使いたくない手法 　ハードウェア(メモリ)の影響を受けやすいため  
　他のnormalization手法  
　LayerNorm … １つの画像に対して正規化(例えばRGBを一緒くたにして正規化)  
　InstanceNorm … １つの画像中の１つのチャンネルに対して正規化  

　解決したいタスクに応じて上記の正規化手法を使い分ける  


Wavenet … convolutionを用いた音声の生成モデル  
　Dilated causal convolution … 層が進むにしたがって畳み込みに使う参照元の間隔(ピクセル距離)を離す．これによってパラメータ数に対する受容野が広くなる(時間的に離れた部分の情報を学習に使う)  

### ・考察(および感想)
試験によく出るらしいので、講義で説明があったもの以外にも有名なモデル(VGGやGANなど)の概要は調査しておく必要がありそうだ  

## Transformer
### ・講義のまとめ
BERT … google翻訳で使われている自然言語処理のニューラルネットワークモデル  

RNN　順番に対しての処理を考慮した再帰的なNNネットワーク  

言語モデル　時刻t-1までの情報で時刻tの事後確率を求めるが目標  
　　　　　　→これで同時確率が計算できる  

RNN ｘ 言語モデル　各地点で次にどの言語が来れば自然(事後確率最大)かを計算  

![image](https://user-images.githubusercontent.com/87635559/126141736-2877bb73-8c38-43dc-9e2d-34984b1c0bfe.png)  

Seq2Seq  
  Encoder-Decoderモデル  
　入力系列をEncoderで内部状態に変換 → 内部状態からDecodeして出力  

![image](https://user-images.githubusercontent.com/87635559/126141852-41c3df5b-5d6a-4ff8-887a-a0730d67b062.png)  

Transformer  
　並列計算が可能なためRNNに比べて計算が高速な上、Self-Attentionと呼ばれる機構を用いることにより、局所的な位置しか参照できないCNNと異なり、系列内の任意の位置の情報を参照することを可能にした  

　Self-attention  
　ニューラル機械翻訳の問題点 … 長さに弱い　翻訳元の文んオ内容を１つのベクトルで表現するため、文が長くなると表現力が足りない  

　Attention … 情報量が多くなってきたときにどこに注意を払うべきかの重要度（重み）を求める手法  
　Attentionは辞書オブジェクト … queryに一致するkeyを検索し、対応するvalueを得る操作であると見做すことが出来る  

![image](https://user-images.githubusercontent.com/87635559/126142054-2167e537-e12f-42ff-b294-85849bdc8889.png)  

Key-Value attention  
Attentionを使うと文が長くなっても精度が落ちにくい  

Transformer (Attention is all you need)  
RNNを使わない，Attentionのみ．RNNを使用したものに比べて計算量が少ない  

![image](https://user-images.githubusercontent.com/87635559/126142154-2c9e684d-3b9b-42cb-ab27-96e3d431493e.png)  


①	RNNを使わないので別に位置情報を保持しておく必要がある  
②	Self-attention  
③	全結合層  
④	 RNNを使わないので順序情報を持たない．そこで未来の単語を見ないようにマスク  

Attentionには2種類ある  
Source Target Attention … 受け取った情報に対して近いものをattentionベクトルとして返す、注目する  
Self-Attention … 自分の入力だけで学習的に注目すべき点を決めていく．CNNに近いイメージ  

### ・実行結果のキャプチャ
BLEUは機械翻訳の分野において最も一般的な自動評価基準の一つで、予め用意した複数の参照訳と、機械翻訳モデルが出力した訳のn-gramのマッチ率に基づく指標．  
プロが翻訳した結果との予測結果の比較．  

■BERT（seq2seq）  

![image](https://user-images.githubusercontent.com/87635559/126142403-8ea6b528-4e41-4265-9ddf-7595cf848dc9.png)  

![image](https://user-images.githubusercontent.com/87635559/126142437-dac55a42-3fd6-44d6-b4fd-eef171b0c35f.png)  

実行がエラー、または予測がUNK(unknown)になってしまった．  
Pytorchの0.4.0を前提としているようだったが，installの実行でエラーになっていたので、それが原因かもしれない．  
手動でcudaなどPytorch 0.4.0のrequireをすべてinstallするのは大変そうなのでここでは深追いしない．  

■BERT (transformer)  
![image](https://user-images.githubusercontent.com/87635559/126142641-b7cd7aca-d48c-4e42-9304-a420c2e21594.png)  
Positional-Encodingの可視化  

![image](https://user-images.githubusercontent.com/87635559/126142705-637d948f-c719-4897-b1cd-d98aa5350668.png)  
![image](https://user-images.githubusercontent.com/87635559/126142755-20602ab0-73c0-471f-a241-c57178770a76.png)  
![image](https://user-images.githubusercontent.com/87635559/126142803-2082373f-32d0-4c11-82f4-676ad4b4024a.png)  

エラーは出ずに学習、推論と処理は流れたが，得られた結果がちょっと芳しくないように見える．(翻訳はうまくいっているようだが…)  
Pytorchのバーションなど前提が異なるのだろうか．  


### ・考察(および感想)
講義中に実装コードまで解説されていたが、自分の知識不足で理解できない部分が多かった．transformerは重要な概念の１つのようなので，動画を見返しつつ，理解を深めておきたい．

### ・追加調査
Transformerの解説  
https://qiita.com/omiita/items/07e69aef6c156d23c538  

BERTの解説  
https://qiita.com/omiita/items/72998858efc19a368e50  

講義では自然言語処理(翻訳)への応用について説明があったが，画像、動画処理へのtransformerの応用も発表されているようだ．  
Transformerを用いることでCNNよりも良い結果が得られたらしい．  
TransformerモデルはCNNのように個人が利用，開発可能なものなのだろうか？  

Vision-Transformer  
(An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)  

・完全に畳み込みなし  
・SoTAを上回る性能を約1/16の計算コストで達成  

Vision Transformerのポイント  
・画像パッチを単語のように扱う  
・アーキテクチャはTransformerのエンコーダー部分  
・巨大なデータセットJFT-300Mで事前学習  

画像に対するattentionの適応  

![image](https://user-images.githubusercontent.com/87635559/126142999-a1e25180-a263-4e42-a38d-1135fac6df8d.png)  

他の手法と比較して精度を更新  
![image](https://user-images.githubusercontent.com/87635559/126143166-a375ad0c-8142-4140-9289-d8cd7efe3195.png)  

上記は分類問題を対象としているが、他の画像系タスクへのtransformerの適応研究、例えばセグメンテーションへ(Vision Transformers for Dense Prediction)や動画(ViViT: A Video Vision Transformer)も発表されているようだ．  

非常に盛り上がっている研究分野のようであり，個人的にも面白そうと感じる分野なので理解を深めてみたい．  

参考および引用元  
https://qiita.com/omiita/items/0049ade809c4817670d7  
https://ai-scholar.tech/articles/transformer/visiontransformer  
https://sorabatake.jp/20454/  
https://cyberagent.ai/blog/research/14721/#paper5  
https://medium.com/axinc/dpt-vision-transformer%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%97%E3%81%9F%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%83%A2%E3%83%87%E3%83%AB-88db4842b4a7  


## 物体検知・セグメンテーション
### ・講義のまとめ
物体認識タスクには４種類ある  
分類 … クラスラベル  
物体検知 … bounding box  
意味領域分割 … 各ピクセルに対して単一のクラスラベル(同じ種類の物体を区別しない)  
個体領域分割 … 各ピクセルに対して単一のクラスラベル(同じ種類の物体を区別する)  

![image](https://user-images.githubusercontent.com/87635559/126143401-f4e1938e-6575-46e5-9bdd-505a6af808bd.png)  

データセット … モデルを評価するためのデータ  
VOC12, ILSVRC17, MS COCO18など多数のデータセットが公開されている  
データセットごとに含まれるクラス数、データ数、Box/画像(一枚の画像にいくつ物体が含まれているか)が異なる  

Box/画像は画像が現実的、日常的なものか、アイコン的なものかを考えるうえで大事な指標  

評価指標  
分類問題における評価指標  
Confusion Matrix（混合指標）  

![image](https://user-images.githubusercontent.com/87635559/126143511-3e19c7d7-2eaa-4183-b55b-aa745a63a5b5.png)  

Precision(適合率) = TP / (TP+FP)　正解と予想したものがどれだけ正しかったか  
Recall (再現性) = TP / (TP + FN)　正解をどれだけ取りこぼしなく予想できたか  
(Accuracy (精度・正解率）= (TP + TN) / (TP+TN+FN+FP)　→　どれだけ正確に予想できているか)  
(Specificity (特異性) = TN / (FP + TN) recallの逆，間違いをどれだけ取りこぼしなく予想できたか)  
PR曲線　Confidence(出力された確率)に対してPositive/Negativeを判断ときの閾値を変更するとPrecision, Recallの値も変わる．それの関係をグラフ化したもの  

![image](https://user-images.githubusercontent.com/87635559/126143672-be75e549-a3c2-44c0-a169-8d891bc7abe8.png)  

IoU  
物体位置の予測精度  
![image](https://user-images.githubusercontent.com/87635559/126143769-d8830e45-dcad-476e-82c3-ccf0729bb4ec.png)  

物体検知ではconfidence, IoU両方を用いて正解、不正解を判別する  

Average Precision　あるクラスに対するPR曲線の下側面積  
Mean Average Precision 全クラス関するAverage Precisionの平均  

検出精度に加え、検出速度も物体検知の評価指標となる  
Frames per Secons (1秒間に処理できるフレーム数)  

深層学習及び物体検出の代表的ネットワーク  
![image](https://user-images.githubusercontent.com/87635559/126143841-3d99efdc-493e-4baa-8711-1908e8f3d296.png)  
緑文字 … ２段階検出器　候補領域の検出とクラス推定を別々に行う 相対的に精度が高いが推論も遅い  
黄色文字 … 1段階検出器　候補領域の検出とクラス推定を同時に行う 相対的に精度が低いが推論が速い  

具体的なモデル紹介  
SSD : Single Shot Mutibox Detector  
1段階検出器, デフォルトボックスを用意しておき、物体に合わせて適応させていく  
VGG16がベースネットワーク  
マルチスケール特徴マップという手法がSSDの特徴  
8732個のデフォルトボックス数を持つ  

多数のデフォルトボックスを用意した弊害  
Non-Maximum Suppression … 冗長な数のバウンディングボックスが作られる　→　最もIoUが大きなボックスを残す  

Hard Negative Mining … 背景として認識されるバウンディングボックスが多数発生  
非背景：背景の比率が1:3になるように背景と判定されたバウンディングボックスを削除  

Sematic Segmentationの概要  
Convolution + poolingの過程で解像度が小さくなっていくがピクセルレベルで物体検知する必要のあるsemantic segmentationでは何とかして元の解像度に戻す必要がある．これをUp-samplingの壁という  

Deconvolution/Transposed convolution  
1.	特徴マップのピクセル感覚をstrideだけあける  
2.	特徴マップの周りに(kernel size -1 ) – paddingだけ余白を作る  
3.	畳み込み演算を行う  

![image](https://user-images.githubusercontent.com/87635559/126144045-d22c10e4-aa9e-4006-b947-51e8758d66b8.png)  

解像度は戻るが、Poolingで失われた情報(≒輪郭などの細かい情報)が復元されるわけではない  

輪郭情報の補間のためにup-samplignしながら同じ解像度の低レイヤーPooling層の出力を加算してローカルな情報の補間を試みている  

U-Net  
Poolingする前の情報を補完しながら解像度を復元するネットワークの代表例  
チェンネル方向への結合が特徴  

別のup-sampling法  
Unpooling … pooling時にどこが最大の値を持っていたかをswitch variablesとして保持しておき，その情報をもとにpooling情報の復元を行う  

### ・考察(および感想)
IoUやup-samplingなど物体検知、セグメンテーションで基礎となる概念を学べてよかった．  
物体検知は数多の研究が発表されているようだが，学んだ基礎概念を踏まえつつ，有名どころだけでも試験までにそれぞれの特徴的を抑えておきたい．  


### ・追加調査
・分類指標についての参考  
https://qiita.com/K5K/items/5da52e99861483cae876  

・YOLO  
物体検知の有名なモデルとして講義でも少し触れられたYOLOについて調べてみた．  
YOLO (You Only Look Once: Unified, Real-Time Object Detection)  
「人類は画像を一目見て，瞬時にそれが画像の中にある物体が何であるのか，どこにあるのか，どのように相互作用しているのかを理解する．」というコンセプトで行われた研究論文．  

オブジェクトの検出とクラス分類の２つのプロセスを同時に行う1段階検出器である．  

入力画像を正方形に分割(grid cell)し，grid cellごとにbounding boxesの推定とどのクラスに属するかの確率を推定し出力(probability map)、それを結合して最終的な出力を得る  

![image](https://user-images.githubusercontent.com/87635559/126144302-952831f6-bece-4639-98cc-0452f75c174f.png)  

特徴  
・シンプルなネットワーク構成で高速  
・背景と物体の区別がしやすい  
・一般化が可能：花などの自然の画像を学習させて，アート作品のような絵の画像でテストした場合，YOLOはDPMやR-CNNよりもはるかに優れている  

ディスアドバンテージ  
・(最先端の手法と比較して)精度が低い  
・小さな物体の検出が困難  

YOLOは年々更新されており，最新のものとしてYOLOv5がある．  

多分試験に出そうな気もするし、有名どころなので(時間があれば論文を読むなどして)もう少し理解を深めた方が良い気がする．  

参考および引用元  
https://www.renom.jp/ja/notebooks/tutorial/image_processing/yolo/notebook.html  
https://qiita.com/cv_carnavi/items/68dcda71e90321574a2b  
https://blog.negativemind.com/2019/02/21/general-object-recognition-yolo/  
https://deepsquare.jp/2020/09/yolo/  
