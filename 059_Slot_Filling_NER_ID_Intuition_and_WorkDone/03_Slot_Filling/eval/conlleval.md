例如数据格式如下：

词\t词性\t命名实体\t正确label\t模型预测的label

![img](../pictures/output_file_format.png)

保证最后一列是模型预测的label，倒数第二列是正确的label

文件名假设为output.txt

那么可以用以下命令评测：

```bash
perl ./conlleval –d "\t" < output.txt
```

-d参数用来指定行之间的分隔符，默认是空格