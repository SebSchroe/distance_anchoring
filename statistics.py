# analyze sample size
import statsmodels.stats.power as smp

effect_size = 0.161 # Cohen's d
alpha = 0.05 # default alpha
power = 0.8 # default power
alternative = 'larger' # can be 'two-sided', 'larger', 'smaller'

analysis = smp.TTestIndPower()

sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)

print(f'Sample size: {sample_size}')