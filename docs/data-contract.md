# 稀疏复核资料与概率定义

labels顺序固定混淆矩阵的行列：行是真类别，列是观察标签。类别先验pi及每reviewer矩阵行各自归一；Dirichlet的归一常数可不计，但全部初值和目标核对使用相同省略。目标为所有非anchor item的log(sum_z pi_z product_r theta_r[z,label])，anchor item改用log(pi_anchor product_r theta_r[anchor,label])，再加先验项sum(alpha-1)log参数。M步pi包含全部item，anchor后验为严格one-hot。reviewer矩阵只累计该reviewer实际给出的标签。

每start给对角值，其余质量均分到其他列，pi均匀；另可接收完整显式初值，须严格正且归一。参数收敛与目标增量容差须同时满足，浮点下降容差按配置单独给定并报告。normal含x5/r4无anchor孤立分量。缺席记录不补票。后验熵为自然对数，anchor熵为零。

已有domain汇总每item的票数和anchor，不建立潜在类别概率，不运行EM。
