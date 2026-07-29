# Mermaid 极简速查



## 1. flowchart 流程图【架构首选】

### 布局方向

LR 左→右｜TD 上→下｜RL 右→左｜BT 下→上

```mermaid
flowchart LR
    A[矩形] --> B[[圆角]]
    B --> C(椭圆)
    C --> D{菱形}
    D --> E[(数据库)]
```

连线：
`A --> B` 实线
`A -.-> B` 虚线
`A --文字--> B` 带标注

子图分组

```mermaid
flowchart LR
subgraph a
    A["A网球"]
    B["B网球"]
end
```

自定义颜色

```mermaid
flowchart LR
A[节点]:::blue
classDef blue fill:#e6f7ff
```

注释：`%% 注释内容`

## 2. sequenceDiagram 时序图（接口交互）

```mermaid
sequenceDiagram
客户端->>服务:请求
服务-->>客户端:响应
```



## 4. stateDiagram-v2 状态机

```mermaid
stateDiagram-v2
[*] --> 待支付
待支付 --> 已支付
```



## 6. classDiagram 类图

```mermaid
classDiagram
class User{
+id:int
+Login()
}
```

4. 复杂C4、gitgraph兼容性差，优先flowchart

## Transformer LR 成品模板

```mermaid
flowchart LR
    S["源序列 Source Tokens"]:::src ---> Encoder[Encoder 多层堆叠]:::enc
    T["目标上文 Target Tokens"]:::tgt --> Decoder[Decoder 多层堆叠]:::dec
    Encoder -.编码上下文.-> Decoder
    Decoder --> Out["预测下一个Token"]:::out

    classDef src fill:#e6f7ff
    classDef enc fill:#b7e1cd
    classDef tgt fill:#fff2cc
    classDef dec fill:#ffe8cc
    classDef out fill:#fce4ec
```

## 花括号

$\underbrace {下方}$



$\overbrace {上方}$



$\left \{ \begin{aligned} 左边 \end{aligned} \right.$     $f(x)=\left\{ \begin{aligned} f(\boldsymbol{x}) &= \boldsymbol{x}^\top A \boldsymbol{x}\\ \boldsymbol{x}&\in\mathbb{R}^n\\ A&\in\mathbb{R}^{n\times n} \end{aligned} \right.$





$$
\left.
\begin{aligned}
f(\boldsymbol{x}) &= \boldsymbol{x}^\top A \boldsymbol{x}\\
\boldsymbol{x}&\in\mathbb{R}^n\\
A&\in\mathbb{R}^{n\times n}
\end{aligned}
\right\} =f(x)
$$


