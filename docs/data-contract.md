# 稀疏复核资料与概率定义

labels顺序固定混淆矩阵的行列：行是真类别，列是观察标签。类别先验pi及每reviewer矩阵行各自归一；Dirichlet的归一常数可不计，但全部初值和目标核对使用相同省略。目标为所有非anchor item的log(sum_z pi_z product_r theta_r[z,label])，anchor item改用log(pi_anchor product_r theta_r[anchor,label])，再加先验项sum(alpha-1)log参数。M步pi包含全部item，anchor后验为严格one-hot。reviewer矩阵只累计该reviewer实际给出的标签。

每start给对角值，其余质量均分到其他列，pi均匀；另可接收完整显式初值，须严格正且归一。参数收敛与目标增量容差须同时满足，浮点下降容差按配置decrease_tolerance单独给定并随结果报告。normal含x5/r4无anchor孤立分量。缺席记录不补票。后验熵为自然对数，anchor熵为零。

analysis_data.domain只做票数与anchor汇总。analysis_data.ds按上述口径运行多初值MAP-EM：E步log-sum-exp归一，M步含anchor计数与alpha-1伪计数；每轮保存目标、参数与最大变化；参数最大变化低于parameter_tolerance且目标增量低于objective_tolerance才报收敛；目标下降超过decrease_tolerance即停止并解释；达到max_iterations不报收敛。各初值结果全部保留，按最终目标择优、同分取编号较小者。reviewer-item二部图连通分量随结果导出，无anchor分量标注类别方向不可靠；潜在类别概率是模型推断而非外部真值，不得回灌计算复核员准确率。`python -m analysis_data.cli objective 资料 参数.json`用给定参数独立重算目标，参数文件可直接取fit输出中某个final。
