import tushare as ts
#使用tushare财经包
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel
import seaborn as sns
import statsmodels.api as sm
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

pro=ts.pro_api('4ebe174595f619b026848f80540c80b87f4267c1e91690852eb3c8b1')
    #这里请替换成自己的api，到tushare个人信息中的接口TOKEN复制
    

def main():
    
    '''
    因为有csv文件，所以第二次运行的时候可以将这两段代码注释了
    '''
    #--------------获取沪深300和A股票的数据---------------------------
    
    # data=pro.query('index_daily',ts_code='000300.SH',start_date='20100101', end_date='20211231')#使用tushare的数据接口获取从10-21的沪深300指数数据
    # #用这个接口还要2000积分QAQ，我花了7块钱租了一个礼拜qaq。
    # hs300_bank=get_300_stock() #获取沪深300当中所有的银行股
    # A_stock=pro.query('daily',ts_code=hs300_bank[0],start_date='20100101', end_date='20211231')#选中第一支银行股，获取其10-21的每日行情
    
    #------------将数据保存为csv格式-------------------
    
    # data.to_csv("沪深300.csv")
    # A_stock.to_csv("A股票.csv")
    
    #----------------------------------------------------
    
    
    
    df_hs300=pd.read_csv('沪深300.csv') 
    A_stock=pd.read_csv('A股票.csv')
    #读取csv文件
    
    #--------------merge合并和聚合-----------------------------
    df=df_hs300.merge(A_stock,how='outer')
    #合并沪深300和A股票
    grouped=df.groupby('ts_code')['close'].agg([np.mean,np.max,np.min,np.median])
    print("合并与聚合")
    print(grouped)
    
    #---------------------画收盘价和开盘价---------------------------------------
    
    draw_300_open_close(df_hs300)
    #时间跨度太长看不清楚收盘价和开盘价的区别，可以一个时间为一个月的图
    draw_300_open_close_short(df_hs300)
    
    #------------------------计算简单收益率-------------------------------------
    
    df_hs300['简单收益率']=df_hs300['close'].pct_change()
    A_stock['简单收益率']=A_stock['close'].pct_change()
    print("简单收益率hs300")
    print(df_hs300['简单收益率'])
    print("简单收益率A")
    print(A_stock['简单收益率'])
    
    #---------------------正太分布置信区间-----------------------------
    
    hs300_mean=df_hs300['pct_chg'].mean()
    h300_std=df_hs300['pct_chg'].std()
    #计算均值和标准差    
    conf=norm.interval(0.95, hs300_mean, h300_std)
    print("正态分布95%置信区间")
    print(conf)
    
    #------------------------单样本t检验-------------------------------
    
    t, p = ttest_1samp(df_hs300['pct_chg'], 0)
    print(f"t = {t}, p = {p}")
    print(f'p值为{p}大于0.05，接受原假设沪深300指数的收益率均值为0')
    
    #-----------------------配对t检验-------------------------
    
    data_clean(df_hs300,A_stock)#数据清洗
    t, p = ttest_rel(df_hs300['pct_chg'], A_stock['pct_chg'])
    print(f"t = {t}, p = {p}")
    print(f'p值为{p}大于0.05接受原假设沪深300和A股票的收益率相等')
    
    #-------------------------相关系数和散点图--------------------
    
    sns.scatterplot(y=df_hs300['pct_chg'],x=A_stock['pct_chg'])  #一个点的散点图
    plt.show()
    draw_scatterplot(df_hs300,A_stock) #画在一起的散点图
    print('相关系数为：')
    print(df_hs300['pct_chg'].corr( A_stock['pct_chg']))
    
    #-------------------------CAPM-------------------------
    
    rf=(1+0.04)**(1/360)-1  #无风险收益率
    ret=pd.merge(df_hs300['pct_chg'],A_stock['pct_chg'],left_index=True,right_index=True,how='inner')  #将沪深300和A股票的收益数据合到一个表格当中，后面的函数需要这样的格式
    Eret=ret-rf  #风险溢价  rm-rf
    
    model_capm=sm.OLS(Eret.pct_chg_y,sm.add_constant(Eret.pct_chg_x)) #调用回归函数
    result=model_capm.fit()
    print(result.summary())
    
    print('其中A股票的β值为0.5472，代表沪深300每涨10%,A股票就上涨5.47%')
        
    
    
    
    
    
    
    
    
    
def get_300_stock():
    hs300=pro.index_weight(index_code='000300.SH', trade_date='20211231')
    #获取沪深300，21年12月31日所包含的所有股票
    
    """
    接口只支持每分钟调取200次，不能过快的调用所以分成3次调用，每次间隔31秒
    """
    hs300_one=hs300[:100]
    hs300_two=hs300[100:200]
    hs300_three=hs300[200:]
    hs300_bank=[]
    
    for index,i in hs300_one['con_code'].items():
        stock_basic=pro.stock_basic(ts_code=i)
        if stock_basic['industry'][0]=='银行':
            hs300_bank.append(stock_basic['ts_code'][0])
            
    print("等待30秒")        
    time.sleep(31)
            
    for index,i in hs300_two['con_code'].items():
        stock_basic=pro.stock_basic(ts_code=i)
        if stock_basic['industry'][0]=='银行':
            hs300_bank.append(stock_basic['ts_code'][0])
            
    print("等待30秒")        
    time.sleep(31)
            
    for index,i in hs300_three['con_code'].items():
        stock_basic=pro.stock_basic(ts_code=i)
        if stock_basic['industry'][0]=='银行':
            hs300_bank.append(stock_basic['ts_code'][0])
    
    return hs300_bank

def draw_300_open_close(df_hs300):
    
    plt.figure()#画板
    
    plt.plot(df_hs300['close'],linewidth=1.0)#画收盘价
    plt.plot(df_hs300['open'],linewidth=0.5)#画开盘价
    
    plt.title("2010-2021沪深300\n开盘价和收盘价曲线图") #标题
    
    plt.xlabel("时间") #轴标题
    plt.ylabel("沪深300指数")
    
    stock_time=['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'] #x坐标轴
    a = [x for x in range(0,len(df_hs300['close']),len(df_hs300['close'])//11)]
    plt.xticks(a,stock_time)
    
    plt.show()         
    
    
def draw_300_open_close_short(df_hs300):
    short_time=df_hs300[df_hs300['trade_date']>20211201]
    plt.plot(short_time['close'],linewidth=1.0)
    plt.plot(short_time['open'],linewidth=1.0)
    plt.title("2021年12月沪深300\n开盘价和收盘价曲线图")
    plt.xlabel("时间")
    stock_time=['12-01','12-11','12-21','12-31']
    a = [x for x in range(0,len(short_time['close']),len(short_time['close'])//3)]
    plt.xticks(a,stock_time)
    plt.ylabel("沪深300指数")
    plt.show()   
    
def data_clean(df_hs300,A_stock):
    '''
    在计算配对t检验之间需要进行一下数据清洗，因为A股票在2010年到2021之间可能存在停盘或者特殊情况造成那天没有数据。
    所以要在沪深300指数的数据中删除对应的那几天，这样才能保持A股票和沪深300指数数据保持一致
    '''
    hs300_data=list(df_hs300['trade_date'])
    stock_data=list(A_stock['trade_date'])
    ret = list(set(hs300_data) ^ set(stock_data))    
    for i in ret:
        df_hs300.drop(df_hs300.index[(df_hs300['trade_date'] == i)], inplace=True)

       
def draw_scatterplot(df_hs300,A_stock):
    x_asix=list(range(0,len(df_hs300)))
    sns.scatterplot(data=df_hs300,y='pct_chg',x=x_asix,alpha=0.2)
    sns.scatterplot(data=A_stock,y='pct_chg',x=x_asix,alpha=0.15)     
    plt.show()
    
    

if __name__ == "__main__":
    main()
