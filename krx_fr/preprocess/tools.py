
import os

import pandas as pd

def read_data_from_folder(folder):
    # 사용할 데이터를 불러오겠습니다 
    data_list = os.listdir(folder)

    result = {}

    for filename in data_list:
        result[filename.split('.')[0]] = pd.read_csv(folder+filename, index_col=0)

    return result

def dict_data_preprocess(input_data_dict:dict, function, window_size="M", fillna=True, start_date="2012-01-01", end_date="2022-12-31"):
    """
    data_list : [list] -> Factor들의 데이터프레임이 담긴 리스트를 받습니다
       column_list : [list] -> data_list의 순서와 일치하는 Factor명의 리스트를 받습니다 
       function : function -> np.mean / np.sum
       window_size : str -> "Y","Q","M" 등 (Default - "Y")
       fillna : bool -> NaN값에 0을 채울지 결정합니다 (Default - True)
    """
    start_index = pd.date_range(start=start_date,end=end_date,freq=window_size+"S")
    end_index = pd.date_range(start=start_date,end=end_date,freq=window_size)
    final_dict = {}
    for start_idx, end_idx in zip(start_index, end_index): #t=0,1...이런 식으로 진행

        column_list = []
        for i, key in enumerate(input_data_dict.keys()):
            column_list.append(key)
            df = input_data_dict[key]
            df.index = pd.to_datetime(df.index)
            if i == 0:
                first_df = function(df.loc[start_idx:end_idx], axis=0)
            elif i == 1:
                concat_df = pd.concat([first_df, function(df.loc[start_idx:end_idx], axis=0)], axis=1, ignore_index=True)
            else:
                concat_df = pd.concat([concat_df, function(df.loc[start_idx:end_idx], axis=0)], axis=1, ignore_index=True)    
        concat_df.columns = column_list
        concat_df.sort_index(inplace=True)
        if fillna == True:
            concat_df = concat_df.fillna(0) # 공시데이터는 회사가 아예 공시한 적이 없어도 NaN값이 뜨기 때문에, 재무데이터를 기준으로 Drop하는 것이 좋다고 판단하여 fillna를 사용함
        final_dict[str(start_idx)[:10]] = concat_df
        
    return final_dict

## factor / factor
# Factor를 나눌때 사용합니다 (ex 매출총이익 / 매출을 구하는 경우)
def dict_data_divide(input_data_dict:dict, divide_what:list, divide_into:str, return_origianl=False):
    '''
    input_data_dict : [dict] -> {t:DataFrame}꼴의 딕셔너리를 받습니다
    divide_what : [list] -> divide_what / divide_into
    divide_into : [str] -> divide_what / divide_into
    return_original : [Boolean] -> divide_what과 divide_into에서 지정된 원본 데이터를 반환 할지 선택합니다
    '''
    return_dict = {}
    for t, df in input_data_dict.items():
        return_df = pd.DataFrame()
        slice_col = divide_what + [divide_into]
        new_col = list(map(lambda x: x + "/" +  divide_into, divide_what))

        return_df[new_col] = df[divide_what].apply(lambda x: x / df[divide_into])
        if return_origianl == False:
            not_select_data = df.drop(slice_col,axis=1)
            return_dict[t] = pd.merge(return_df, not_select_data,left_index=True,right_index=True)
        else: # 원본데이터도 함께 반환하는 경우...
            return_dict[t] = pd.merge(return_df, df,left_index=True,right_index=True)
        
    return return_dict

## factor + factor를 계산합니다
def dict_data_plus(input_data_dict:dict, plus_one:str, plus_two:str, return_origianl=False):
    '''
    input_data_dict : [dict] -> {t:DataFrame}꼴의 딕셔너리를 받습니다
    plus_one + plus_two -> 을 계산합니다
    return_original : [Boolean] -> divide_what과 divide_into에서 지정된 원본 데이터를 반환 할지 선택합니다
    '''
    return_dict = {}
    for t, df in input_data_dict.items():
        return_df = pd.DataFrame()
        slice_col = [plus_one] + [plus_two]

        new_col = "{}+{}".format(plus_one,plus_two)
        return_df[new_col] = df[plus_one] + df[plus_two]

        if return_origianl == False:
            not_select_data = df.drop(slice_col,axis=1)
            return_dict[t] = pd.merge(return_df, not_select_data,left_index=True,right_index=True)
        else: # 원본데이터도 함께 반환하는 경우...
            return_dict[t] = pd.merge(return_df, df,left_index=True,right_index=True)
        
    return return_dict

# dict data에서 컬럼 드랍하는 용도
# {t:DataFrame}꼴의 딕셔너리의 특정 컬럼 또는 NaN 값을 drop하는데 사용합니다
def dict_data_drop(input_data_dict:dict, drop_col:list=[], dropna=False):
    '''
    input_data_dict : [dictionary] -> {t:dataframe}꼴의 딕셔너리를 받습니다
    drop_col : [list] -> drop해줄 컬럼의 리스트를 받습니다
    dropna : [Boolean] -> Default = False (중요!!! dropna=True이면 drop_col=[]로 (빈리스트) 주어줘야 합니다!!!)
    '''
    return_dict = {}
    for t, df in input_data_dict.items():
        if dropna == True:
            return_dict[t] = df.dropna(axis=0)
        else:
            return_dict[t] = df.drop(drop_col, axis=1)
    return return_dict

# 스케일링 함수
# {t:DataFrame}꼴의 딕셔너리를 원하는 스케일러로 스케일링하기
def dict_data_scale(input_data_dict:dict, scaler):
    '''
    input_data_dict : [dictionary] -> {t:dataframe}꼴의 딕셔너리를 받습니다
    scaler : [function] -> (pp.zscore / pp.rank / pp.minmax / pp.quartile)
    '''
    return_dict = {}
    for t, df in input_data_dict.items():
        return_dict[t] = scaler(df)
    return return_dict

# 공시랑 재무 합치기
# 공시 딕셔너리와 재무 딕셔너리를 하나의 딕셔너리로 합칩니다
def dict_data_concat(report_dict: dict, financial_dict:dict, dropna=True):
    '''report_dict, financial_dict : [dictionary] -> preprocess_report_data 함수를 통과시킨 딕셔너리를 주어줘야 합니다
       dropna -> 데이터 중 하나라도 NaN 값이 존재하는 기업은 drop시킵니다 (Default=True)
    '''
    return_dict = {}
    for key, report in report_dict.items():
        financial = financial_dict[key]
        report_col, financial_col = report_dict[key].columns.to_list(), financial_dict[key].columns.to_list()
        concat_key = report_col + financial_col
        concat_df = pd.concat([report, financial], axis=1, ignore_index=True).sort_index()
        if dropna == True:
            concat_df = concat_df.dropna().sort_index()
        concat_df.columns = concat_key
        return_dict[key] = concat_df
    return return_dict