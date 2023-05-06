# zhongda
用不同的方式对池沸腾问题代码项目进行时间上的优化，这其中包含
0、原始代码-由CPU进行深度学习训练

以下为优化代码

1、使用GPU进行训练(使用.cuda())

2、使用GPU进行分布式训练(使用Dataparallel())

以下开始考虑对分布式训练中任务在不同GPU的资源占用进行调整

3、在双GPU服务器上对任务使用的GPU0的batchsize进行调整(对Dataparallel()进行了修改)
