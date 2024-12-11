import matplotlib.pyplot as plt
import numpy as np

result = {'NoC@85': 1.7681159420289845, 'NoC@90': 1.808695652173912, 'NoC@95': 1.4666666666666661, 'iou_series': np.array([0.7803515 , 0.86046649, 0.90223006, 0.9154424 , 0.91927357,
       0.92156636, 0.92313267, 0.92438088, 0.92535841, 0.92642567,
       0.92712607, 0.92772381, 0.92820591])}

result_extrapolation = {'NoC@85': 1.8318840579710147, 'NoC@90': 1.8753623188405795, 'NoC@95': 1.3855072463768106, 'iou_series': np.array([0.77968536, 0.8594294 , 0.89876414, 0.91238155, 0.91896791,
       0.92164   , 0.92426669, 0.92627327, 0.92754933, 0.92844749,
       0.92950903, 0.93028417, 0.93068716])}
result_extrapolation_biased = {'NoC@85': 1.886956521739131, 'NoC@90': 1.823188405797101, 'NoC@95': 1.3999999999999995, 'iou_series': np.array([0.77845772, 0.85802633, 0.89696378, 0.91048114, 0.91795871,
       0.921035  , 0.92298979, 0.9246153 , 0.9264829 , 0.92735134,
       0.92830448, 0.92905782, 0.92978372])}

plt.style.use('ggplot')
plt.plot(np.arange(1, 13+1), result['iou_series'], '.-', label='IoU (original)')
plt.plot(np.arange(1, 13+1), result_extrapolation['iou_series'], '.-', label='IoU (ex.)')
plt.plot(np.arange(1, 13+1), result_extrapolation_biased['iou_series'], '.--', label='IoU (ex. biased)')
plt.legend()
plt.title('IoU')
plt.savefig('Figure_3.svg')