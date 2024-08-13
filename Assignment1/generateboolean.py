import pandas as pd
import numpy as np

for INPUTS in range(1,5):
    #makes Inputs Matrix 
    auxiliary = [[0],[1]]
    if INPUTS ==1:
        input_dataset = auxiliary
    else:
        for _ in range(INPUTS-1):
            input_dataset = []
            for out in auxiliary:
                point = out.copy()
                point.append(0)
                input_dataset.append(point)
                point = out.copy()
                point.append(1)
                input_dataset.append(point)
            auxiliary = input_dataset.copy()

    #Makes Function Vectors
    auxiliary = [[0],[1]]
    for _ in range(pow(2,INPUTS)-1):
        function_dataset = []
        for out in auxiliary:
            point = out.copy()
            point.append(0)
            function_dataset.append(point)
            point = out.copy()
            point.append(1)
            function_dataset.append(point)
        auxiliary = function_dataset.copy()
        

    input_np = np.array(input_dataset)
    dataset = []
    for func in function_dataset:
        function_np = np.array(func)
        point = np.hstack([input_np,function_np.reshape(-1,1)])
        dataset.append(point)
    dataset_np = np.vstack(dataset)

    columns = ['X'+str(i) for i in range(INPUTS)]
    columns.append("Y")
    df = pd.DataFrame(dataset_np, columns=columns)
    print(df)

    # Dump the DataFrame to a CSV file
    df.to_csv('./csv/Inputs'+str(INPUTS)+'.csv',index=False)

