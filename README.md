NYCU-DLP-LAB 1
===
最後編輯時間
>2023 / 08 / 08

作者
> 王語 , 312605003 

文件說明
---
`code`
：Python程式碼

`report`：結報

`output`：Python程式的紀錄檔，記錄該次執行所使用的訓練參數以及模型設定。

`Fig`：結報中使用到的圖片

lab1.py
---
一個 two layer neural network ，使用反向傳播進行訓練，可以更改相關參數觀察訓練結果。

本程式可透過 command line arguments 控制訓練參數以及模型設定，詳細說明請透過下列命令：

```console
$ /...直譯器路徑.../python3.9 /...程式所在路徑.../lab1.py -h
```

每次執行結束會生成一個output.txt檔案，紀錄該次執行之參數，此txt檔會存放在當前所在路徑。

lab1_conv.py
---
加分題之程式，是一個使用卷積層對假想輸入資料進行結果預測的神經網路。