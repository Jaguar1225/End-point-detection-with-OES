import matplotlib.pyplot as plt

class PlotStyle:    
    """OriginLab 스타일의 플로팅을 위한 스타일 설정"""
    
    @staticmethod
    def set_style():
        """기본 스타일 설정"""
        # 기본 스타일 설정
        plt.style.use('default')
        
        # 폰트 설정
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 25
        plt.rcParams['axes.titlesize'] = 30
        plt.rcParams['axes.labelsize'] = 25
        
        # 그리드 설정
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.5
        
        # 축 설정
        plt.rcParams['axes.linewidth'] = 3.0
        plt.rcParams['axes.edgecolor'] = 'black'
        
        # 틱 설정
        plt.rcParams['xtick.major.width'] = 3.0
        plt.rcParams['ytick.major.width'] = 3.0
        plt.rcParams['xtick.minor.width'] = 3.0
        plt.rcParams['ytick.minor.width'] = 3.0
        
        # 여백 설정
        plt.rcParams['figure.autolayout'] = True
        plt.rcParams['figure.constrained_layout.use'] = True