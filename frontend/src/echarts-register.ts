import {use} from 'echarts/core';
import {CanvasRenderer} from 'echarts/renderers';
import {LineChart, ScatterChart} from 'echarts/charts';
import {DataZoomComponent, GridComponent, TooltipComponent} from 'echarts/components';

use([CanvasRenderer, LineChart, ScatterChart, GridComponent, TooltipComponent, DataZoomComponent]);
