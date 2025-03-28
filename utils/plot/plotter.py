import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Union, Optional
import os
from datetime import datetime
from .style import PlotStyle

class Plotter:
    """플로팅 유틸리티 클래스"""
    
    def __init__(self, save_dir: str = datetime.now().strftime('%Y%m%d')):
        """
        Args:
            save_dir (str): 그래프 저장 디렉토리
        """
        self.save_dir = f'plots/{save_dir}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.markers = ['o', 's', 'D', 'P', '*', 'X', 'd']
        self.linestyles = ['-', '--', '-.', ':']
            

    def plot_heatmap(self,
                    data: Union[List, np.ndarray],
                    title: str = '',
                    xlabel: str = '',
                    ylabel: str = '',
                    save_name: Optional[str] = None,
                    dpi: int = 300) -> None:
        """
        히트맵 플로팅
        
        Args:
            data (Union[List, np.ndarray]): 플로팅할 데이터
            title (str): 그래프 제목
            xlabel (str): x축 레이블
            ylabel (str): y축 레이블
            save_name (Optional[str]): 저장 파일명
            dpi (int): 저장 해상도
        """
        plt.figure(figsize=(16, 12))
        plt.imshow(data, aspect='auto', cmap='RdBu_r')
        plt.colorbar()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_line(self, 
                 x: Union[List, np.ndarray],
                 y: Union[List, np.ndarray],
                 title: str = '',
                 xlabel: str = '',
                 ylabel: str = '',
                 color: str = None,
                 marker: str = None,
                 linestyle: str = None,
                 label: str = None,
                 grid: bool = True,
                 save_name: Optional[str] = None,
                 dpi: int = 300) -> None:
        """
        선 그래프 생성
        
        Args:
            x: x축 데이터
            y: y축 데이터
            title: 그래프 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            color: 선 색상
            marker: 마커 스타일
            linestyle: 선 스타일
            label: 범례 레이블
            grid: 그리드 표시 여부
            save_name: 저장 파일명
            dpi: 해상도
        """
        plt.figure(figsize=(8, 6))

        if isinstance(x, list):
            x = np.array(x).T
        if len(x.shape) == 1:
            x = x[:,np.newaxis]
                
        if isinstance(y, list):
            y = np.array(y).T
        if len(y.shape) == 1:
            y = y[:,np.newaxis]
        
        if color is None:
            color = self.colors[0:x.shape[-1]]  # 단일 색상으로 수정
        if marker is None:
            marker = self.markers[0:x.shape[-1]]
        if linestyle is None:
            linestyle = self.linestyles[0:x.shape[-1]]
        if label is None:
            label = [f'{i}' for i in range(x.shape[-1])]
        
        for i in range(x.shape[-1]):
            plt.plot(x[:,i], y[:,i], color=color[i], marker=marker[i], linestyle=linestyle[i], 
                label=label[i], linewidth=2.5, markersize=6)
                
        plt.title(title, pad=15)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.ylim(0.5, 1.2)

        if grid:
            plt.grid(True, linestyle='--', alpha=0.3)
            
        if label:
            plt.legend()
            
        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    
    def plot_scatter(self,
                    x: Union[List, np.ndarray],
                    y: Union[List, np.ndarray],
                    title: str = '',
                    xlabel: str = '',
                    ylabel: str = '',
                    color: str = None,
                    marker: str = None,
                    label: str = None,
                    grid: bool = True,
                    save_name: Optional[str] = None,
                    dpi: int = 300) -> None:
        """
        산점도 생성
        
        Args:
            x: x축 데이터
            y: y축 데이터
            title: 그래프 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            color: 점 색상
            marker: 마커 스타일
            label: 범례 레이블
            grid: 그리드 표시 여부
            save_name: 저장 파일명
            dpi: 해상도
        """

        if isinstance(x, list):
            x = np.array(x).T
        if len(x.shape) == 1:
            x = x[:,np.newaxis]
                
        if isinstance(y, list):
            y = np.array(y).T
        if len(y.shape) == 1:
            y = y[:,np.newaxis]

        plt.figure(figsize=(8, 6))
        
        if color is None:
            color = self.colors[0:x.shape[-1]]
        if marker is None:
            marker = self.markers[0:x.shape[-1]]
        if label is None:
            label = [f'{i}' for i in range(x.shape[-1])]
            
        for i in range(x.shape[-1]):
            plt.scatter(x[i], y[i], color=color[i], marker=marker[i], label=label[i], s=50)
        
        plt.title(title, pad=15)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if grid:
            plt.grid(True, linestyle='--', alpha=0.3)
            
        if label:
            plt.legend()
            
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    
    def plot_comparison(self,
                       true_values: Union[List, np.ndarray],
                       predicted_values: Union[List, np.ndarray],
                       title: str = 'True vs Predicted Values',
                       xlabel: str = 'True Values',
                       ylabel: str = 'Predicted Values',
                       color: str = None,
                       marker: str = None,
                       grid: bool = True,
                       save_name: Optional[str] = None,
                       dpi: int = 300) -> None:
        """
        예측값과 실제값 비교 그래프 생성
        
        Args:
            true_values: 실제값
            predicted_values: 예측값
            title: 그래프 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            color: 점 색상
            marker: 마커 스타일
            grid: 그리드 표시 여부
            save_name: 저장 파일명
            dpi: 해상도
        """
        plt.figure(figsize=(8, 6))
        
        if color is None:
            color = self.colors[0]
        if marker is None:
            marker = self.markers[0]
            
        plt.scatter(true_values, predicted_values, color=color, marker=marker, 
                   alpha=0.5, s=50, label='Data points')
        
        # 대각선 추가
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Perfect prediction')
        
        plt.title(title, pad=15)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if grid:
            plt.grid(True, linestyle='--', alpha=0.3)
            
        plt.legend()
        plt.tight_layout()

        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
