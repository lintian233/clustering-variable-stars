import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy import signal
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.signal import find_peaks
import numpy as np
from scipy.signal import correlate
import statsmodels.api as sm
from statsmodels import robust
import statsmodels.stats.stattools as stats
from scipy.optimize import curve_fit
from scipy.linalg import eigh
import concurrent.futures as cf
class OlgeFeatures:
    def __init__(self,path) -> None:
        self.file_path = path
        self.bin_index = None
        self.time = None
        self.mag = None
        self.frequency = np.linspace(0.0003, 24, 800000)
        self.power = None
        self.peaks = None
        self.features = None

    def init(self):
        self.time, self.mag = self._read_table()
        self.power = self._lomb_scargle()
        self._allis_filter()
        freq_index = self._get_indexof_log_span(self.frequency,10)
        freq_index.insert(0, 0)
        self.peaks = self._find_peak_of_index(self.power,freq_index)
        self.bin_index = self._bin_split()
        self.features = self._generate_features_90()

        return self.features
    
    def _read_table(self):
        data = pd.read_table(self.file_path,sep = '\\s+',names=['time','mag','err'])
        return data.iloc[:,0],data.iloc[:,1]

    def _lomb_scargle(self):
        return LombScargle(self.time, self.mag).power(self.frequency)

    def _find_first_greater(self, array, value):
        left = 0
        right = len(array) - 1
        while left <= right:
            mid = (left + right) // 2
            if array[mid] <= value:
                left = mid + 1
            else:
                right = mid - 1
        return left
        
    def _allis_filter(self):
        pass

    def _get_indexof_log_span(self, data, bin):
        log_end = np.log10(data[-1] - data[1000])
        log_start =np.log10(data[1000])
        log_step = (log_end - log_start) / bin

        value_arr = np.array([(10 ** (log_start + i * log_step)) for i in range(1,bin + 1)])
        index_arr = []
        
        for i in value_arr:
            index_arr.append(self._find_first_greater(data,i))
        return index_arr
    
    def _find_peak_of_index(self,data, index_arr): 
        peaks = np.array([])
        for i in range(len(index_arr) - 1):
            start = index_arr[i]
            end = index_arr[i + 1]
            peak,_ = find_peaks(data[start:end],distance=(end - start) ** 0.5)
            peak += start
            peaks = np.concatenate((peaks,peak))
        return peaks
    
    def _bin_split(self):
        freq_idx_1 = self._find_first_greater(self.frequency, 1)
        index_of_1_log_arr = self._get_indexof_log_span(self.frequency[0:freq_idx_1], 7)
        index_of_1_log_arr.insert(0, 0)
        index_of_1_log_arr = np.array(index_of_1_log_arr)
        index_freq_1dot5_to_24 = self._find_first_greater(self.frequency, 1.5)
        index_of_arr = np.linspace(index_freq_1dot5_to_24,len(self.frequency), 11)
        index_of_arr = np.int_(index_of_arr)
        return np.concatenate((index_of_1_log_arr,index_of_arr))
        

    def _normalized_iqr(self,array):
        q1 = np.percentile(array, 25)
        q3 = np.percentile(array, 75)
        iqr = q1 - q3
        med = np.median(array)
        if (iqr == 0):
            iqr = 1
        normalized_arr = (array - med) / iqr
        return normalized_arr
    
    def _generate_features_90(self):
        featrues = np.zeros(90,dtype=np.float64)        
        f_idx = 0
        bins_idx = self.bin_index
        peaks = self.peaks
        power = self.power

        for i in range(len(bins_idx) - 1):
            current_peak = np.int_(peaks[(peaks > bins_idx[i]) & (peaks < bins_idx[i + 1])])
            current_power = power[current_peak]
            current_power = np.sort(np.log(current_power))[::-1][:5]
            current_power = self._normalized_iqr(current_power)
            featrues[f_idx:f_idx + len(current_power)] = current_power
            f_idx += len(current_power)
        return featrues
    
class AstroDataFeatures:
    def __init__(self, path) -> None:
        self.path = path
        self.time = None
        self.mag = None
        # self.periodogram_features = None
        # self.huber_mean = None
        # self.mad = None
        # self.iqr_q31 = None
        # self.q31_robust_mean = None
        # self.q955_minus = None
        # self.range_cumsum = None
        # self.percentage_beyond_sigma = None
        # self.percent_of_meadian_range_1_over_10 = None
        # self.largest_percentage_difference = None
        # self.mag_3rd_fraction = None
        # self.median_based_skew = None
        # self.flux_percentile_ratio_mid20 = None
        # self.flux_percentile_ratio_mid35 = None
        # self.flux_percentile_ratio_mid50 = None
        # self.flux_percentile_ratio_mid65 = None
        # self.flux_percentile_ratio_mid80 = None
        # self.robust_kurtosis = None
        # self.robust_measure_of_left_weight = None
        # self.robust_measure_of_right_weight = None
        # self.robust_kurtosis_excess = None
        # self.abbe = None
        # self.regularity_of_lc_50 = None
        # self.regularity_of_lc_100 = None
        # self.regularity_of_lc_250 = None
        # self.autocorrelation = None
        # self.stetson_k_for_autocor = None
        # self.stetson_k_for_mag = None
        # self.quality_fit_x2qso_v = None
        # self.rcs_to_phase_lc = None
        # self.p2p_scatter = None
        # self.variability_index_of_lc = None
        # self.logp = None
        # self.ampij
        # self.phij = None
        # self.log10_harmonic_amplitude_ratio = None
        # self.log10_fit_residual_raw_iqr_lc = None
        # self.abbe_of_harmonic_fit_lc = None
        self.features = None

    def thread_worker(self, func, *args):
        if len(args) == 0:
            return func()
        
        result = func(*args)
        return result

    def INIT(self):

        """
        INIT all features
        """
        self.time, self.mag = self._read_table()

        #单线程版本:
        periodogram_features = self._generate_periodogram_features()
        huber_mean = self._huber_mean(self.time, self.mag)
        mad = self._mad(self.mag)
        iqr_q31 = self._iqr_q31(self.mag)
        q31_robust_mean = self._q31_robust_mean(self.time, self.mag)
        q955_minus = self._q955_minus(self.mag)
        range_cumsum = self._range_cumsum(self.mag)
        percentage_beyond_sigma = self._percentage_beyond_sigma(self.mag)
        percent_of_meadian_range_1_over_10 = self._percent_of_meadian_range_1_over_10(self.mag)
        largest_percentage_difference = self._largest_percentage_difference(self.mag)
        mag_3rd_fraction = self._mag_3rd_fraction(self.mag)
        median_based_skew = self._median_based_skew(self.mag)
        flux_percentile_ratio_mid20 = self._flux_percentile_ratio_mid20(self.mag)
        flux_percentile_ratio_mid35 = self._flux_percentile_ratio_mid35(self.mag)
        flux_percentile_ratio_mid50 = self._flux_percentile_ratio_mid50(self.mag)
        flux_percentile_ratio_mid65 = self._flux_percentile_ratio_mid65(self.mag)
        flux_percentile_ratio_mid80 = self._flux_percentile_ratio_mid80(self.mag)
        robust_kurtosis = self._robust_kurtosis(self.mag)
        robust_kurtosis_excess = self._robust_kurtosis_excess(self.mag)
        abbe = self._abbe(self.mag)
        regularity_of_lc_50 = self._regularity_of_lc(self.time, self.mag, 50)
        regularity_of_lc_100 = self._regularity_of_lc(self.time, self.mag, 100)
        regularity_of_lc_250 = self._regularity_of_lc(self.time, self.mag, 250)
        autocorrelation = self._autocorrelation(self.mag)
        stetson_k_for_autocor = self._stetson_k_for_autocor(autocorrelation)
        stetson_k_for_mag = self._stetson_k_for_mag(self.mag)
        rcs_to_phase_lc = self._rcs_to_phase_lc(self.time, self.mag)
        p2p_scatter = self._p2p_scatter(self.time, self.mag)
        variability_index_of_lc = self._variability_index_of_lc(self.time, self.mag)
        logp = self._logP(self.time, self.mag)

        ampij, phij, harmonic_mag, fitting_mag= self._amp_ph_of_harmonic_fit_lc(self.time, self.mag)
        self.log10_harmonic_amplitude_ratio = self._log10_harmonic_amplitude_ratio(ampij)
        self.log10_fit_residual_raw_iqr_lc = self._log10_fit_residual_raw_iqr_lc(self.mag, fitting_mag)
        self.abbe_of_harmonic_fit_lc = self._abbe_of_residuals_har_fit(self.mag, harmonic_mag, fitting_mag)




        # 创建线程池
        # with cf.ThreadPoolExecutor(max_workers = 4 ) as executor:
        #     periodogram_features = executor.submit(self.thread_worker, self._generate_periodogram_features)
        #     huber_mean = executor.submit(self.thread_worker, self._huber_mean, self.time, self.mag)
        #     mad = executor.submit(self.thread_worker, self._mad, self.mag)
        #     iqr_q31 = executor.submit(self.thread_worker, self._iqr_q31, self.mag)
        #     q31_robust_mean = executor.submit(self.thread_worker, self._q31_robust_mean, self.time, self.mag)
        #     q955_minus = executor.submit(self.thread_worker, self._q955_minus, self.mag)
        #     range_cumsum = executor.submit(self.thread_worker, self._range_cumsum, self.mag)
        #     percentage_beyond_sigma = executor.submit(self.thread_worker, self._percentage_beyond_sigma, self.mag)
        #     percent_of_meadian_range_1_over_10 = executor.submit(self.thread_worker, self._percent_of_meadian_range_1_over_10, self.mag)
        #     largest_percentage_difference = executor.submit(self.thread_worker, self._largest_percentage_difference, self.mag)
        #     mag_3rd_fraction = executor.submit(self.thread_worker, self._mag_3rd_fraction, self.mag)
        #     median_based_skew = executor.submit(self.thread_worker, self._median_based_skew, self.mag)
        #     flux_percentile_ratio_mid20 = executor.submit(self.thread_worker, self._flux_percentile_ratio_mid20, self.mag)
        #     flux_percentile_ratio_mid35 = executor.submit(self.thread_worker, self._flux_percentile_ratio_mid35, self.mag)
        #     flux_percentile_ratio_mid50 = executor.submit(self.thread_worker, self._flux_percentile_ratio_mid50, self.mag)
        #     flux_percentile_ratio_mid65 = executor.submit(self.thread_worker, self._flux_percentile_ratio_mid65, self.mag)
        #     flux_percentile_ratio_mid80 = executor.submit(self.thread_worker, self._flux_percentile_ratio_mid80, self.mag)
        #     robust_kurtosis = executor.submit(self.thread_worker, self._robust_kurtosis, self.mag)
        #     robust_kurtosis_excess = executor.submit(self.thread_worker, self._robust_kurtosis_excess, self.mag)
        #     abbe = executor.submit(self.thread_worker, self._abbe, self.mag)
        #     regularity_of_lc_50 = executor.submit(self.thread_worker, self._regularity_of_lc, self.time, self.mag, 50)
        #     regularity_of_lc_100 = executor.submit(self.thread_worker, self._regularity_of_lc, self.time, self.mag, 100)
        #     regularity_of_lc_250 = executor.submit(self.thread_worker, self._regularity_of_lc, self.time, self.mag, 250)
        #     autocorrelation = executor.submit(self.thread_worker, self._autocorrelation, self.mag)
        #     stetson_k_for_mag = executor.submit(self.thread_worker, self._stetson_k_for_mag, self.mag)
        #     rcs_to_phase_lc = executor.submit(self.thread_worker, self._rcs_to_phase_lc, self.time, self.mag)
        #     p2p_scatter = executor.submit(self.thread_worker, self._p2p_scatter, self.time, self.mag)
        #     variability_index_of_lc = executor.submit(self.thread_worker, self._variability_index_of_lc, self.time, self.mag)
        #     logp = executor.submit(self.thread_worker, self._logP, self.time, self.mag)
        #     fit = executor.submit(self.thread_worker, self._amp_ph_of_harmonic_fit_lc, self.time, self.mag)
        
        # periodogram_features = periodogram_features.result()
        # huber_mean = huber_mean.result()
        # mad = mad.result()
        # iqr_q31 = iqr_q31.result()
        # q31_robust_mean = q31_robust_mean.result()
        # q955_minus = q955_minus.result()
        # range_cumsum = range_cumsum.result()
        # percentage_beyond_sigma = percentage_beyond_sigma.result()
        # percent_of_meadian_range_1_over_10 = percent_of_meadian_range_1_over_10.result()
        # largest_percentage_difference = largest_percentage_difference.result()
        # mag_3rd_fraction = mag_3rd_fraction.result()
        # median_based_skew = median_based_skew.result()
        # flux_percentile_ratio_mid20 = flux_percentile_ratio_mid20.result()
        # flux_percentile_ratio_mid35 = flux_percentile_ratio_mid35.result()
        # flux_percentile_ratio_mid50 = flux_percentile_ratio_mid50.result()
        # flux_percentile_ratio_mid65 = flux_percentile_ratio_mid65.result()
        # flux_percentile_ratio_mid80 = flux_percentile_ratio_mid80.result()
        # robust_kurtosis = robust_kurtosis.result()
        # robust_kurtosis_excess = robust_kurtosis_excess.result()
        # abbe = abbe.result()
        # regularity_of_lc_50 = regularity_of_lc_50.result()
        # regularity_of_lc_100 = regularity_of_lc_100.result()
        # regularity_of_lc_250 = regularity_of_lc_250.result()
        # autocorrelation = autocorrelation.result()
        # stetson_k_for_autocor = self._stetson_k_for_autocor(autocorrelation)
        # stetson_k_for_mag = stetson_k_for_mag.result()
        # rcs_to_phase_lc = rcs_to_phase_lc.result()
        # p2p_scatter = p2p_scatter.result()
        # variability_index_of_lc = variability_index_of_lc.result()
        # logp = logp.result()
        # ampij = fit.result()[0]
        # phij = fit.result()[1]
        # fitting_mag = fit.result()[2]
        # harmonic_mag = fit.result()[3]


        log10_harmonic_amplitude_ratio = self._log10_harmonic_amplitude_ratio(ampij)
        log10_fit_residual_raw_iqr_lc = self._log10_fit_residual_raw_iqr_lc(self.mag, fitting_mag)
        abbe_of_harmonic_fit_lc = self._abbe_of_residuals_har_fit(self.mag, harmonic_mag, fitting_mag)

        self._quality_fit_x2qso_v()
        self._robust_measure_of_left_weight(self.mag)
        self._robust_measure_of_right_weight(self.mag)

        self.features = np.concatenate(([huber_mean['mag']], 
                                        [mad], 
                                        [iqr_q31], 
                                        q31_robust_mean, 
                                        [q955_minus], 
                                        [range_cumsum], 
                                        [percentage_beyond_sigma], 
                                        [percent_of_meadian_range_1_over_10], 
                                        [largest_percentage_difference], 
                                        [mag_3rd_fraction], 
                                        [median_based_skew], 
                                        [flux_percentile_ratio_mid20], 
                                        [flux_percentile_ratio_mid35], 
                                        [flux_percentile_ratio_mid50], 
                                        [flux_percentile_ratio_mid65], 
                                        [flux_percentile_ratio_mid80], 
                                        [robust_kurtosis],
                                        [robust_kurtosis_excess], 
                                        [abbe], 
                                        [regularity_of_lc_50], 
                                        [regularity_of_lc_100], 
                                        [regularity_of_lc_250], 
                                        [stetson_k_for_autocor], 
                                        [stetson_k_for_mag], 
                                        [rcs_to_phase_lc], 
                                        [p2p_scatter], 
                                        [variability_index_of_lc], 
                                        [logp],
                                        ampij,
                                        phij , 
                                        log10_harmonic_amplitude_ratio[0],
                                        log10_harmonic_amplitude_ratio[1], 
                                        [log10_fit_residual_raw_iqr_lc], 
                                        abbe_of_harmonic_fit_lc,
                                        periodogram_features
                                        ))
                                        
        return self.features

    def _read_table(self):
        """
        Read the data from the file
        file format: object.dat time mag err
        return: time, mag
        """
        data = pd.read_table(self.path,sep='\\s+',names=['time','mag','err'])
        return data.iloc[:,0],data.iloc[:,1]
    

    def _generate_periodogram_features(self):
        """
        The first 5 maximums of the log-power for each the 18 bins
        return: Periodogram nth maximums (90)
        """
        return OlgeFeatures(self.path).init()
    
    def _huber_mean(self, time, mag):
        """
        Robust Mean measure of the magnitudes based on Huber’s M-estimation
        基于Huber的M估计的星等的稳健均值测量
        return: hubermean
        """
        rlm_model = sm.RLM(time, mag, M=sm.robust.norms.HuberT())
        rlm_results = rlm_model.fit()
        return rlm_results.params
    
    def _mad(self, mag):
        """
        Median absolute deviation of the magnitudes
        星等的中值绝对偏差
        return: mad
        """
        return robust.mad(mag)
    
    def _iqr_q31(self, mag):
        """
        Difference between the 75th and 25th percentiles of the magnitudes
        星等的四分位距
        return: iqr
        """
        q1 = np.percentile(mag, 25)
        q3 = np.percentile(mag, 75)
        return q3 - q1
    
    def _q31_robust_mean(self, time, mag):
        """
        Ratio of Q31 to the Robust Mean magnitude
        Q31 与 robust mean 的星等比值
        return: q31_robust_mean
        """
        ratio = self._iqr_q31(mag) / self._huber_mean(time, mag)
        return ratio
    
    def _q955_minus(self, mag):
        """
        The median of the magnitudes over the 95th percentile minus
        the median of the magnitudes under the 5th percentile
        return: q95 - q5
        """
        mag_95 = np.percentile(mag, 95)
        mag_5 = np.percentile(mag, 5)
        median_95 = np.median(mag[mag > mag_95])
        median_5 = np.median(mag[mag < mag_5])
        Amplitude = median_95 - median_5
        return Amplitude
    
    def _range_cumsum(self, mag):
        """
        The cumulative sum of the magnitudes over the range of the magnitudes
        星等累积和的范围
        https://ui.adsabs.harvard.edu/abs/2014A%26A...566A..43K/abstract
        return: range_cumsum
        """
        def spec_cumsum(x):
            return np.cumsum((x - np.mean(x)) / (np.std(x) * len(x)))
        
        return np.ptp(spec_cumsum(mag))
    
    def _percentage_beyond_sigma(self, mag):
        """
        #Percentage of points beyond 1sigma from the weighted mean
        1sigma 以外的点的百分比
        return: percentage_beyond_sigma
        """
        weighted_mean = np.average(mag, weights=None)
        std_dev = np.sqrt(np.average((mag - weighted_mean)**2, weights=None))
        beyond_sigma = np.sum(np.abs(mag - weighted_mean) > std_dev)
        percentage = beyond_sigma / len(mag) * 100

        return percentage
    

    def _percent_of_meadian_range_1_over_10(self, mag):
        """
        Fraction of points within a tenth of the magnitude range of the median magnitude
        在中位数范围的十分之一的点的星等百分比
        return: percentage
        """
        mag_sort = np.sort(mag)
        mag_range = np.ptp(mag_sort)
        mag_median = np.median(mag_sort)
        mag_range_tenth = mag_range / 10
        mag_median_range = mag_median + mag_range_tenth
        mag_median_range_count = np.sum((mag_sort > mag_median - mag_range_tenth) & (mag_sort < mag_median_range))
        mag_median_range_fraction = mag_median_range_count / len(mag_sort)
        return mag_median_range_fraction
    
    def _largest_percentage_difference(self, mag):
        """
        Largest percentage difference between either 
        the maximum or minimum magnitude and the median
        最大或最小星等与中位数之间的最大百分比差异
        return largest_percentage_difference
        """
        median = np.median(mag)  
        max_val = np.max(mag)  
        min_val = np.min(mag)  

        diff1 = abs(max_val - median) / median * 100  
        diff2 = abs(min_val - median) / median * 100  

        return max(diff1, diff2) 
    
    def _mag_3rd_fraction(self,mag):
        """
        The fraction of points over the 3rd quartile plus 1.5 times 
        the interquartile range of the magnitude
        星等的四分位数加上1.5倍的四分位数范围的点的星等百分比
        this work
        return: mag_3rd_fraction
        """
        mag_sort = np.sort(mag)
        q3 = np.percentile(mag_sort, 75)
        q1 = np.percentile(mag_sort, 25)
        iqr = q3 - q1
        mag_3rd = q3 + 1.5 * iqr
        mag_3rd_count = np.sum(mag_sort > mag_3rd)
        mag_3rd_fraction = mag_3rd_count / len(mag_sort)
        
        return mag_3rd_fraction
    
    def _median_based_skew(self,arr):
        """
        Median based measure of the skew
        基于中位数的偏度测量
        return: skew
        """
        median = np.median(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        skew = 3 * (mean - median) / std
        return skew
    
    def _flux_percentile_ratio_mid20(self,arr):
        """
        Flux Percentile Ratio Mid-20: Sorted flux percentile ratio 𝐹40,60/𝐹5,95
        return: flux_percentile_ratio_mid20
        """
        arr_sort = np.sort(arr)
        arr_5 = np.percentile(arr_sort, 5)
        arr_95 = np.percentile(arr_sort, 95)
        arr_40 = np.percentile(arr_sort, 40)
        arr_60 = np.percentile(arr_sort, 60)
        return (arr_40 - arr_60) / (arr_5 - arr_95)

    def _flux_percentile_ratio_mid35(self,arr):
        """
        Flux Percentile Ratio Mid-35: Sorted flux percentile ratio 𝐹32.5,67.5/𝐹5,95
        return: flux_percentile_ratio_mid35
        """
        arr_sort = np.sort(arr)
        arr_5 = np.percentile(arr_sort, 5)
        arr_95 = np.percentile(arr_sort, 95)
        arr_32_5 = np.percentile(arr_sort, 32.5)
        arr_67_5 = np.percentile(arr_sort, 67.5)
        return (arr_32_5 - arr_67_5) / (arr_5 - arr_95)

    def _flux_percentile_ratio_mid50(self, arr):
        """
        Flux Percentile Ratio Mid-50: Sorted flux percentile ratio 𝐹25,75/𝐹5,95
        return: flux_percentile_ratio_mid50
        """
        arr_sort = np.sort(arr)
        arr_5 = np.percentile(arr_sort, 5)
        arr_95 = np.percentile(arr_sort, 95)
        arr_25 = np.percentile(arr_sort, 25)
        arr_75 = np.percentile(arr_sort, 75)
        return (arr_25 - arr_75) / (arr_5 - arr_95)

    def _flux_percentile_ratio_mid65(self, arr):
        """
        Flux Percentile Ratio Mid-65: Sorted flux percentile ratio 𝐹17.5,82.5/𝐹5,95
        return: flux_percentile_ratio_mid65
        """
        arr_sort = np.sort(arr)
        arr_5 = np.percentile(arr_sort, 5)
        arr_95 = np.percentile(arr_sort, 95)
        arr_17_5 = np.percentile(arr_sort, 17.5)
        arr_82_5 = np.percentile(arr_sort, 82.5)
        return (arr_17_5 - arr_82_5) / (arr_5 - arr_95)

    def _flux_percentile_ratio_mid80(self, arr):
        """
        Flux Percentile Ratio Mid-80: Sorted flux percentile ratio 𝐹10,90/𝐹5,95
        return: flux_percentile_ratio_mid80
        """
        arr_sort = np.sort(arr)
        arr_5 = np.percentile(arr_sort, 5)
        arr_95 = np.percentile(arr_sort, 95)
        arr_10 = np.percentile(arr_sort, 10)
        arr_90 = np.percentile(arr_sort, 90)
        return (arr_10 - arr_90) / (arr_5 - arr_95)
    
    def _robust_kurtosis(self, mag):
        """
        Robust kurtosis measure based on Stetson variability index
        基于Stetson可变性指数的健壮峰度测量
        kurtosis_measures是一个包含四个峰度测量值的元组:
        kr1: 标准峰度估计器
        kr2: 基于八分位数的峰度估计器
        kr3: 基于超过期望值的峰度估计器
        kr4: 基于高低分位数之间的差距的峰度测量
        return: robust_kurtosis
        """
        return stats.robust_kurtosis(mag)[0]
    
    def _robust_measure_of_left_weight(self, mag):
        """
        Robust measure of the left weight of the distribution
        分布左侧权重的健壮测量
        return: robust_measure_of_left_weight
        """
        pass

    def _robust_measure_of_right_weight(self,mag):
        """
        Robust measure of the right weight of the distribution
        分布右侧权重的健壮测量
        return: robust_measure_of_right_weight
        """
        pass
    def _robust_kurtosis_excess(self, mag):
        """
        Robust measure of kurtosis based on on exceedance expectations
        基于超出期望值的健壮峰度测量
        return: robust_kurtosis_excess
        """
        return stats.robust_kurtosis(mag)[2]
    
    def _abbe(self, y):
        """
        Abbe value for the y
        #Measure of the smoothness of the light curve
        光曲线平滑度的度量
        https://arxiv.org/abs/1406.7785v1
        return: abbe
        """
        n = len(y)
        y_bar = np.mean(y)
        sum_diff_sq = np.sum(np.diff(y)**2)
        sum_var_sq = np.sum((y - y_bar)**2)
        A = (n / (2*(n-1))) * (sum_diff_sq / sum_var_sq)
        return A
    
    def __excess_abbe(self, x, y, window_size):
        """
        Excess Abbe Value for the y
        y的超额阿贝值
        https://arxiv.org/abs/1406.7785v1
        return: excess_abbe
        """
        first = np.where(x >= x[0] + 0.5 * window_size)[0][0]
        end = np.where(x <= x[len(x) - 1] - 0.5 * window_size)[0][-1]
        abbe_value = np.zeros(end - first + 1)
        for i in range(first, end):
            indexes = np.where((x > x[i] - 0.5 * window_size) & (x < x[i] + 0.5 * window_size))
            if(len(indexes[0]) < 5):
                continue    
            y_window = y[indexes[0]]
            abbe_value[i - first] = self._abbe(y_window)
        ab_remove_zero = abbe_value[abbe_value != 0]
        abbe_mean = np.mean(ab_remove_zero)
        return abbe_mean - self._abbe(y)

    
    def _regularity_of_lc(self, time, mag, window_size):
        """
        Estimation of the regularity of the light curve variability pattern for window size 50 days
        估计窗口大小为 w 的光曲线变化模式的规律性
        Excess Abbe Value Tsub = 50 d/ 100 d/ 250 d
        https://arxiv.org/abs/1406.7785v1
        """
        return self.__excess_abbe(time, mag, window_size)
    

    def _autocorrelation(self, mag):
        """
        Robust autocorrelation function length for irregular time series
        不规则时间序列的健壮自相关函数
        https://arxiv.org/abs/1212.2398
        Slotted autocorrelation function 
        槽自相关函数长度
        """
        autocorr = sm.tsa.acf(mag, nlags=len(mag))
        return autocorr
    
    def _stetson_k(self, y):
        """
        https://iopscience.iop.org/article/10.1086/133808
        return: stetson_k
        """
        N = len(y)
        sigma = ( ( N / (N - 1) ) ** 0.5 ) * ( y - np.mean(y) ) / np.std(y)
        K = ( 1 / N ) * np.sum(np.abs(sigma)) / ( ( 1 / N ) * np.sum( sigma ** 2 ) ) ** 0.5
        return K

    def _stetson_k_for_autocor(self,autocorr):
        """
        StetsonK applied over the slotted autocorrelation function
        应用于槽自相关函数的StetsonK
        https://iopscience.iop.org/article/10.1086/133808
        """
        return self._stetson_k(autocorr)
    
    def _stetson_k_for_mag(self,mag):
        """
        StetsonK applied over the magnitudes
        应用于星等的StetsonK
        https://iopscience.iop.org/article/10.1086/133808
        """
        return self._stetson_k(mag)
    
    def _quality_fit_x2qso_v(self):
        """
        Quality of fit 𝜒2QSO/v for a quasar-like source, assuming 𝑚𝑎𝑔 = 19
        质量拟合𝜒2QSO/𝜈为类星体源，假设𝑚𝑎𝑔 = 19
        https://ui.adsabs.harvard.edu/abs/2011AJ....141...93B/abstract
        --------不懂
        
        Natural logarithm of expected 𝜒2QSO/v for non-QSO variable
        非QSO变量的预期𝜒2QSO/𝜈的自然对数
        Base 10 logarithm of the period
        --------不懂
        """
        pass

    def __folded_lc(self, time, mag):
        """
        phase folded light curve
        https://ui.adsabs.harvard.edu/abs/2014A%26A...566A..43K/abstract
        """
        frequency = np.linspace(0.0003, 24, 800000)
        power = LombScargle(time, mag).power(frequency)
        best_f = frequency[np.argmax(power)]

        # print("best_f:{}".format(best_f))
        
        period = 1 / best_f
        phase = np.mod(time, period) / period
        
        #phase = (time / period) % 1

        sorted_indices = np.argsort(phase)
        sorted_phase = phase[sorted_indices]
        sorted_mag = mag[sorted_indices]
        return sorted_phase, sorted_mag
    

    def _rcs_to_phase_lc(self, time, mag):
        """
        RCS applied to the phase-folded light curve
        应用于相位折叠光曲线的RCS
        https://ui.adsabs.harvard.edu/abs/2014A%26A...566A..43K/abstract
        """
        sorted_phase, sorted_mag = self.__folded_lc(time, mag)

        def spec_cumsum(x):
            return np.cumsum((x - np.mean(x)) / (np.std(x) * len(x)))
        
        rcs_phase = spec_cumsum(sorted_mag)
        rcs  = np.ptp(rcs_phase)

        return rcs
    
    def _p2p_scatter(self, time, mag):
        """
        https://arxiv.org/abs/1101.2406
        P2p scatter: P/raw : median of the absolute values of the differences
        etween successive magnitudes in the folded light curve normalized by the Median 
        Absolute Deviation (MAD) around the median of the raw light curve.
        P2p scatter: P/raw : 折叠光曲线中连续幅度之间的差值的绝对值的中位数，
        归一化为原始光曲线的中位数绝对偏差(MAD)。
        """
        phase_folded_mag = self.__folded_lc(time, mag)[1]

        #calculate_median_of_differences
        # 计算连续幅度之间的差值
        differences = np.diff(phase_folded_mag)
        
        # 计算差值的绝对值
        absolute_differences = np.abs(differences)
        
        # 计算差值的中位数
        median_of_differences = np.median(absolute_differences)
        
        # 计算原始光曲线的中位数
        median = np.median(mag)
        
        # 计算原始光曲线的中位数绝对偏差（MAD）
        mad = np.median(np.abs(mag - median))
        
        # 归一化为MAD
        normalized_median = median_of_differences / mad

        return normalized_median
    
    def _variability_index_of_lc(self, time, mag):
        """
        Variability index 𝜂𝑒 applied to the the folded light curve
        应用于折叠光曲线的可变性指数𝜂𝑒
        https://arxiv.org/abs/1101.2406
        https://ui.adsabs.harvard.edu/abs/2014A%26A...566A..43K/abstract
        """
        def get_gama(x):
            N = len(x)
            return (1 / (N - 1) * np.std(x) ** 2) * np.sum( np.diff(x) ** 2 )
        
        folded_mag = self.__folded_lc(time, mag)[1]
        return get_gama(folded_mag)
    
    def __whiten(self, x, y):
        """
        Whiten the data
        白化数据
        """
        # 去中心化
        x_ = x - np.mean(x)
        y_ = y - np.mean(y)

        #尺度标准化
        x_ = x_ / np.std(x_)
        y_ = y_ / np.std(y_)
        
        # 计算数据的协方差矩阵
        covariance_matrix = np.cov(x_, y_)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = eigh(covariance_matrix)

        # 计算白化矩阵
        whitening_matrix = np.dot(np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues))), eigenvectors.T)

        # 对原始数据进行白化
        whitened_data = np.dot(whitening_matrix, np.vstack((x_, y_)))

        # 提取白化后的时间和数据
        return whitened_data[0], whitened_data[1]
    
    def _logP(self, time, mag):
        """
        Log10 of the period
        return: log10(1 / best_f)
        """
        frequency = np.linspace(0.0003, 24, 800000)
        power = LombScargle(time, mag).power(frequency)
        best_f = frequency[np.argmax(power)]
        return np.log10(1 / best_f)
    
    def _amp_ph_of_harmonic_fit_lc(self, time, mag):
        """
        Amplitudes of the jth harmonic of the ith period
        https://arxiv.org/abs/1101.2406
        ith 周期的 jth 谐波的振幅
        Phases of the jth harmonic of the ith period remmaped to be between -pi and +pi
        重新映射为-pi和+pi之间的ith周期的jth谐波的相位
        return: second_Aij(8), second_Phij(7), fitting_mag, harmonic_mag
        """
        frequency = np.linspace(0.0003, 24, 800000)
        power = LombScargle(time, mag).power(frequency)
        best_f = frequency[np.argmax(power)]
        def harmonic_func(x, a1, a2, a3, a4, b1, b2, b3, b4, b0):
            a = np.array([a1, a2, a3, a4])
            b = np.array([b1, b2, b3, b4])
            y = np.zeros(len(x))
            for i in range(4):
                y += a[i] * np.sin(2 * np.pi * best_f * (i + 1) * x) + b[i] * np.cos(2 * np.pi * best_f * (i + 1) * x)
            y += b0
            return y

        popt, pcov = curve_fit(harmonic_func, time, mag,maxfev = 1000000)

        y_fit = harmonic_func(time, *popt)
        whitten_mag = self.__whiten(time, mag)[1]
        y_new = y_fit - whitten_mag
        new_power = LombScargle(time, y_new).power(frequency)
        new_best_f = frequency[np.argsort(new_power)[-2:]]
        def best_func(x,*params):

            def _harmonic_func(x, a1, a2, a3, a4, b1, b2, b3, b4, best_f, b0):
                a = np.array([a1, a2, a3, a4])
                b = np.array([b1, b2, b3, b4])
                y = np.zeros(len(x))
                for i in range(4):
                    y += a[i] * np.sin(2 * np.pi * best_f * (i + 1) * x) + b[i] * np.cos(2 * np.pi * best_f * (i + 1) * x)
                y += b0
                return y
        
            a = params[:8]
            b = params[8:16]
            b0 = params[16]
            y = np.zeros(len(x))
            for i in range(2):
                ai = a[i * 4: i * 4 + 4]
                bi = b[i * 4: i * 4 + 4]
                y += _harmonic_func(x, *ai, *bi, new_best_f[i], b0)
            return y + b0
        
        init_params = np.zeros(17)
        new_popt, new_pcov = curve_fit(best_func, time, mag,maxfev = 1000000, p0=init_params)
        
        Ampij = (new_popt[:8] ** 2 + new_popt[8:16] ** 2) ** 0.5
        Phij = np.arctan2(new_popt[:8], new_popt[8:16])
        return Ampij, Phij, best_func(time, *new_popt), harmonic_func(time, *popt)
    
    def _log10_harmonic_amplitude_ratio(self, Ampij):
        """
        Logarithm in base 10 of the amplitude ratios of the 
        jth harmonic with respecto to the 1st amplitude
        Ratio of the amplitude of the second harmonic to the amplitude of the first harmonic
        Log10(第jth谐波振幅与第i谐波振幅的比值)
        https://arxiv.org/abs/1101.2406
        return: harmonic_amplitude_ratio
        """
        logr1 = np.log10(Ampij[1:4] / Ampij[0])
        logr2 = np.log10(Ampij[5:8] / Ampij[4])

        return logr1, logr2
    
    def _log10_fit_residual_raw_iqr_lc(self, mag, fitting_mag):
        """
        Logarithm in base 10 of ratio between the IQR of the residuals of the fit periodic model 
        and the IQR of the raw magnitudes
        拟合周期模型的残差的IQR与原始星等的IQR之间的比率的以10为底的对数
        """
        residuals = mag - fitting_mag
        sorted_residuals = np.sort(residuals)
        residuals_iqr = np.percentile(sorted_residuals, 75) - np.percentile(sorted_residuals, 25)
        sorted_mag  = np.sort(mag)
        mag_iqr = np.percentile(sorted_mag, 75) - np.percentile(sorted_mag, 25)

        logIQR = np.log10(residuals_iqr / mag_iqr)

        return logIQR
    
    def _abbe_of_residuals_har_fit(self,mag, harmonics_mag, fitting_mag):
        """
        Abbe value of the residuals from the Fourier model subtraction of the first and second period
        第一和第二周期的傅里叶模型减法的残差的Abbe值
        """
        first_residuals = mag - harmonics_mag
        second_residuals = mag - fitting_mag
        fr_abbe = self._abbe(first_residuals)
        sr_abbe = self._abbe(second_residuals)
        fs_abbe = (fr_abbe, sr_abbe)
        return fs_abbe
    

