import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class DataDecimator:
    """
    Advanced data decimation analysis and processing for robot sensor data.
    
    Uses multiple criteria to determine optimal decimation factors:
    - Spectral analysis (Nyquist criterion)
    - Information theory (entropy preservation)
    - Correlation analysis
    - Signal reconstruction fidelity
    - Computational efficiency trade-offs
    """
    
    def __init__(self):
        self.original_data = {}
        self.analysis_results = {}
        self.recommended_factors = {}
        
    def load_data(self):
        """Load all CSV data files."""
        print("Loading data files...")
        
        try:
            self.original_data['ref'] = np.genfromtxt('ref.csv', delimiter=',')
            self.original_data['odo'] = np.genfromtxt('odo.csv', delimiter=',') 
            self.original_data['laser'] = np.genfromtxt('laser.csv', delimiter=',')
            
            print(f"Loaded data shapes:")
            for name, data in self.original_data.items():
                print(f"  {name}: {data.shape}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
        return True
    
    def analyze_temporal_characteristics(self, data, name):
        """
        Analyze temporal characteristics of the data.
        
        Args:
            data: Input data array
            name: Name of the dataset
        
        Returns:
            Dictionary with temporal analysis results
        """
        print(f"\nAnalyzing temporal characteristics of {name}...")
        
        results = {}
        
        # For multi-dimensional data, analyze each dimension
        if data.ndim == 1:
            signals = [data]
            signal_names = [name]
        else:
            signals = [data[:, i] for i in range(data.shape[1])]
            signal_names = [f"{name}_dim_{i}" for i in range(data.shape[1])]
        
        for signal_data, signal_name in zip(signals, signal_names):
            signal_results = {}
            
            # 1. Autocorrelation analysis
            autocorr = np.correlate(signal_data, signal_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find correlation length (where autocorr drops to e^-1)
            correlation_threshold = 1/np.e
            correlation_length = np.where(autocorr < correlation_threshold)[0]
            if len(correlation_length) > 0:
                signal_results['correlation_length'] = correlation_length[0]
            else:
                signal_results['correlation_length'] = len(autocorr)
            
            # 2. Spectral analysis
            n_samples = len(signal_data)
            freqs = fftfreq(n_samples, d=1.0)  # Assuming unit sampling frequency
            fft_vals = np.abs(np.array(fft(signal_data)))
            
            # Find dominant frequencies (above 5% of max power)
            power_threshold = 0.05 * np.max(fft_vals)
            dominant_freq_indices = np.where(fft_vals > power_threshold)[0]
            dominant_freqs = freqs[dominant_freq_indices]
            
            signal_results['dominant_frequencies'] = dominant_freqs[:n_samples//2]  # Only positive freqs
            signal_results['max_frequency'] = np.max(np.abs(dominant_freqs[:n_samples//2]))
            
            # 3. Signal complexity (approximate entropy)
            def approximate_entropy(data, m=2, r=None):
                """Calculate approximate entropy."""
                if r is None:
                    r = 0.2 * np.std(data)
                
                N = len(data)
                
                def _maxdist(xi, xj, m):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _phi(m):
                    patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
                    C = np.zeros(N - m + 1)
                    
                    for i in range(N - m + 1):
                        template = patterns[i]
                        matches = sum([1 for pattern in patterns if _maxdist(template, pattern, m) <= r])
                        C[i] = matches / (N - m + 1.0)
                    
                    return np.mean(np.log(C))
                
                return _phi(m) - _phi(m + 1)
            
            try:
                signal_results['approximate_entropy'] = approximate_entropy(signal_data)
            except:
                signal_results['approximate_entropy'] = 0
            
            # 4. Change detection (variance of differences)
            diff_signal = np.diff(signal_data)
            signal_results['change_variance'] = np.var(diff_signal)
            signal_results['change_mean'] = np.mean(np.abs(diff_signal))
            
            results[signal_name] = signal_results
        
        return results
    
    def spectral_decimation_analysis(self, data, name, max_factor=50):
        """
        Determine decimation factor based on spectral content.
        
        Args:
            data: Input data
            name: Dataset name
            max_factor: Maximum decimation factor to consider
        
        Returns:
            Recommended decimation factor based on Nyquist criterion
        """
        print(f"Performing spectral analysis for {name}...")
        
        if data.ndim > 1:
            # For multi-dimensional data, analyze the dimension with highest frequency content
            max_freqs = []
            for i in range(data.shape[1]):
                signal_data = data[:, i]
                freqs = fftfreq(len(signal_data), d=1.0)
                fft_vals = np.abs(np.array(fft(signal_data)))
                
                # Find 99th percentile frequency (effective bandwidth)
                power = fft_vals[:len(fft_vals)//2]**2
                cumulative_power = np.cumsum(power) / np.sum(power)
                freq_99 = freqs[np.where(cumulative_power > 0.99)[0][0]] if np.any(cumulative_power > 0.99) else freqs[len(freqs)//4]
                max_freqs.append(abs(freq_99))
            
            effective_max_freq = max(max_freqs)
        else:
            signal_data = data
            freqs = fftfreq(len(signal_data), d=1.0)
            fft_vals = np.abs(np.array(fft(signal_data)))
            
            power = fft_vals[:len(fft_vals)//2]**2
            cumulative_power = np.cumsum(power) / np.sum(power)
            freq_99 = freqs[np.where(cumulative_power > 0.99)[0][0]] if np.any(cumulative_power > 0.99) else freqs[len(freqs)//4]
            effective_max_freq = abs(freq_99)
        
        # Nyquist criterion: sampling frequency should be at least 2x highest frequency
        # After decimation by factor N, new sampling freq = original_freq / N
        # So: original_freq / N >= 2 * effective_max_freq
        # Therefore: N <= original_freq / (2 * effective_max_freq)
        
        # Assuming original sampling frequency of 1 Hz (unit frequency)
        nyquist_factor = int(1.0 / (2 * effective_max_freq)) if effective_max_freq > 0 else max_factor
        nyquist_factor = max(1, min(nyquist_factor, max_factor))
        
        return nyquist_factor, effective_max_freq
    
    def information_preserving_decimation(self, data, name, max_factor=50):
        """
        Determine decimation factor based on information preservation.
        
        Args:
            data: Input data
            name: Dataset name
            max_factor: Maximum decimation factor to consider
            
        Returns:
            Recommended decimation factor based on information theory
        """
        print(f"Performing information-theoretic analysis for {name}...")
        
        # Test different decimation factors
        factors = range(1, min(max_factor, len(data)//10) + 1)
        information_losses = []
        reconstruction_errors = []
        
        for factor in factors:
            # Decimate data
            decimated = data[::factor]
            
            # Reconstruct using interpolation
            original_indices = np.arange(len(data))
            decimated_indices = np.arange(0, len(data), factor)
            
            if data.ndim == 1:
                reconstructed = np.interp(original_indices, decimated_indices, decimated)
                
                # Calculate reconstruction error
                reconstruction_error = mean_squared_error(data, reconstructed)
                reconstruction_errors.append(reconstruction_error)
                
                # Calculate information loss (entropy difference)
                # Discretize for entropy calculation
                bins = min(50, len(np.unique(data))//2)
                if bins < 2:
                    bins = 2
                    
                orig_hist, _ = np.histogram(data, bins=bins, density=True)
                decimated_hist, _ = np.histogram(decimated, bins=bins, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                orig_entropy = entropy(orig_hist + epsilon)
                decimated_entropy = entropy(decimated_hist + epsilon)
                
                info_loss = np.abs(orig_entropy - decimated_entropy) / orig_entropy if orig_entropy > 0 else 0
                information_losses.append(info_loss)
                
            else:
                # Multi-dimensional data
                total_error = 0
                total_info_loss = 0
                
                for dim in range(data.shape[1]):
                    dim_data = data[:, dim]
                    dim_decimated = decimated[:, dim]
                    
                    reconstructed_dim = np.interp(original_indices, decimated_indices, dim_decimated)
                    total_error += mean_squared_error(dim_data, reconstructed_dim)
                    
                    # Information loss calculation
                    bins = min(50, len(np.unique(dim_data))//2)
                    if bins < 2:
                        bins = 2
                        
                    orig_hist, _ = np.histogram(dim_data, bins=bins, density=True)
                    decimated_hist, _ = np.histogram(dim_decimated, bins=bins, density=True)
                    
                    epsilon = 1e-10
                    orig_entropy = entropy(orig_hist + epsilon)
                    decimated_entropy = entropy(decimated_hist + epsilon)
                    
                    info_loss = np.abs(orig_entropy - decimated_entropy) / orig_entropy if orig_entropy > 0 else 0
                    total_info_loss += info_loss
                
                reconstruction_errors.append(total_error / data.shape[1])
                information_losses.append(total_info_loss / data.shape[1])
        
        # Find optimal factor: minimize weighted combination of information loss and reconstruction error
        reconstruction_errors = np.array(reconstruction_errors)
        information_losses = np.array(information_losses)
        
        # Normalize metrics to [0,1] range
        if np.max(reconstruction_errors) > 0:
            norm_recon_errors = reconstruction_errors / np.max(reconstruction_errors)
        else:
            norm_recon_errors = reconstruction_errors
            
        if np.max(information_losses) > 0:
            norm_info_losses = information_losses / np.max(information_losses)
        else:
            norm_info_losses = information_losses
        
        # Combined metric (equal weighting)
        combined_metric = 0.5 * norm_recon_errors + 0.5 * norm_info_losses
        
        # Find factor with acceptable loss (< 10% combined metric)
        acceptable_factors = [factors[i] for i, loss in enumerate(combined_metric) if loss < 0.1]
        
        if acceptable_factors:
            optimal_factor = max(acceptable_factors)  # Highest acceptable decimation
        else:
            # If no factor meets criteria, choose the best one
            optimal_factor = factors[np.argmin(combined_metric)]
        
        return optimal_factor, reconstruction_errors, information_losses
    
    def correlation_based_decimation(self, data, name):
        """
        Determine decimation factor based on temporal correlation.
        """
        print(f"Analyzing temporal correlation for {name}...")
        
        if data.ndim == 1:
            signal_data = data
        else:
            # Use the dimension with most variation
            variances = np.var(data, axis=0)
            max_var_dim = np.argmax(variances)
            signal_data = data[:, max_var_dim]
        
        # Calculate autocorrelation
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find where autocorrelation drops below threshold
        thresholds = [0.9, 0.8, 0.7, 0.5, 1/np.e]
        correlation_lengths = {}
        
        for threshold in thresholds:
            drop_indices = np.where(autocorr < threshold)[0]
            if len(drop_indices) > 0:
                correlation_lengths[threshold] = drop_indices[0]
            else:
                correlation_lengths[threshold] = len(autocorr)
        
        # Conservative estimate: use 50% correlation threshold
        recommended_factor = max(1, correlation_lengths[0.5] // 2)
        
        return recommended_factor, correlation_lengths
    
    def analyze_all_datasets(self):
        """Perform comprehensive analysis on all datasets."""
        print("=== COMPREHENSIVE DECIMATION ANALYSIS ===\n")
        
        if not self.load_data():
            return
        
        for name, data in self.original_data.items():
            print(f"\n{'='*60}")
            print(f"ANALYZING DATASET: {name.upper()}")
            print(f"Shape: {data.shape}, Size: {data.size} elements")
            print(f"{'='*60}")
            
            analysis = {}
            
            # 1. Temporal characteristics
            temporal_results = self.analyze_temporal_characteristics(data, name)
            analysis['temporal'] = temporal_results
            
            # 2. Spectral analysis
            spectral_factor, max_freq = self.spectral_decimation_analysis(data, name)
            analysis['spectral_factor'] = spectral_factor
            analysis['max_frequency'] = max_freq
            
            # 3. Information preservation
            info_factor, recon_errors, info_losses = self.information_preserving_decimation(data, name)
            analysis['info_factor'] = info_factor
            analysis['reconstruction_errors'] = recon_errors
            analysis['information_losses'] = info_losses
            
            # 4. Correlation analysis
            corr_factor, corr_lengths = self.correlation_based_decimation(data, name)
            analysis['correlation_factor'] = corr_factor
            analysis['correlation_lengths'] = corr_lengths
            
            self.analysis_results[name] = analysis
            
            # Determine final recommendation
            factors = [spectral_factor, info_factor, corr_factor]
            
            # Conservative approach: use minimum of all factors for safety
            conservative_factor = min(factors)
            
            # Balanced approach: use median
            balanced_factor = int(np.median(factors))
            
            # Aggressive approach: use maximum (most decimation)
            aggressive_factor = max(factors)
            
            self.recommended_factors[name] = {
                'conservative': conservative_factor,
                'balanced': balanced_factor, 
                'aggressive': aggressive_factor,
                'individual_factors': {
                    'spectral': spectral_factor,
                    'information': info_factor,
                    'correlation': corr_factor
                }
            }
            
            print(f"\nRECOMMENDATIONS FOR {name}:")
            print(f"  Spectral analysis: {spectral_factor}")
            print(f"  Information preservation: {info_factor}")
            print(f"  Correlation analysis: {corr_factor}")
            print(f"  Conservative (min): {conservative_factor}")
            print(f"  Balanced (median): {balanced_factor}")
            print(f"  Aggressive (max): {aggressive_factor}")
    
    def visualize_analysis(self):
        """Create comprehensive visualization of the analysis results."""
        print("\nGenerating analysis visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        dataset_names = list(self.analysis_results.keys())
        
        # Plot 1: Decimation factors comparison
        ax = axes[0]
        factors_data = []
        labels = []
        
        for name in dataset_names:
            factors = self.recommended_factors[name]['individual_factors']
            factors_data.append([factors['spectral'], factors['information'], factors['correlation']])
            labels.append(name.upper())
        
        factors_array = np.array(factors_data).T
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, factors_array[0], width, label='Spectral', alpha=0.8)
        ax.bar(x, factors_array[1], width, label='Information', alpha=0.8)
        ax.bar(x + width, factors_array[2], width, label='Correlation', alpha=0.8)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Recommended Decimation Factor')
        ax.set_title('Decimation Factor Analysis by Method')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Final recommendations
        ax = axes[1]
        rec_types = ['Conservative', 'Balanced', 'Aggressive']
        rec_data = []
        
        for rec_type in rec_types:
            rec_data.append([self.recommended_factors[name][rec_type.lower()] for name in dataset_names])
        
        rec_array = np.array(rec_data).T
        
        for i, rec_type in enumerate(rec_types):
            ax.bar([f"{name}\n{rec_type}" for name in labels], rec_array[:, i], alpha=0.7, label=rec_type)
        
        ax.set_ylabel('Decimation Factor')
        ax.set_title('Final Recommendations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Plot 3-5: Data characteristics for each dataset
        for i, (name, analysis) in enumerate(self.analysis_results.items()):
            if i >= 3:
                break
                
            ax = axes[i + 2]
            
            # Plot original vs decimated data sample
            data = self.original_data[name]
            factor = self.recommended_factors[name]['balanced']
            
            if data.ndim == 1:
                sample_range = slice(0, min(1000, len(data)))
                ax.plot(data[sample_range], 'b-', alpha=0.7, label='Original', linewidth=1)
                ax.plot(range(0, len(data[sample_range]), factor), 
                       data[sample_range][::factor], 'ro-', markersize=3, 
                       label=f'Decimated (factor {factor})')
            else:
                sample_range = slice(0, min(1000, data.shape[0]))
                # Plot first dimension
                ax.plot(data[sample_range, 0], 'b-', alpha=0.7, label='Original', linewidth=1)
                ax.plot(range(0, len(data[sample_range, 0]), factor),
                       data[sample_range, 0][::factor], 'ro-', markersize=3,
                       label=f'Decimated (factor {factor})')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Signal Value')
            ax.set_title(f'{name.upper()} - Original vs Decimated')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('decimation_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_decimated_files(self, strategy='balanced'):
        """
        Generate decimated versions of all data files.
        
        Args:
            strategy: 'conservative', 'balanced', or 'aggressive'
        """
        print(f"\nGenerating decimated files using {strategy} strategy...")
        
        decimation_summary = []
        
        for name, data in self.original_data.items():
            factor = self.recommended_factors[name][strategy]
            
            # Decimate the data
            decimated_data = data[::factor]
            
            # Save to file
            output_filename = f"{name}_decimated_{strategy}_f{factor}.csv"
            np.savetxt(output_filename, decimated_data, delimiter=',', fmt='%.6f')
            
            # Calculate compression ratio
            original_size = data.shape[0]
            decimated_size = decimated_data.shape[0]
            compression_ratio = original_size / decimated_size
            
            summary_info = {
                'dataset': name,
                'original_size': original_size,
                'decimated_size': decimated_size,
                'decimation_factor': factor,
                'compression_ratio': compression_ratio,
                'filename': output_filename
            }
            
            decimation_summary.append(summary_info)
            
            print(f"  {name}: {original_size} -> {decimated_size} samples "
                  f"(factor {factor}, {compression_ratio:.1f}x compression)")
        
        # Save summary report
        summary_df = pd.DataFrame(decimation_summary)
        summary_df.to_csv(f'decimation_summary_{strategy}.csv', index=False)
        
        print(f"\nDecimation complete! Summary saved to 'decimation_summary_{strategy}.csv'")
        
        return decimation_summary
    
    def print_comprehensive_report(self):
        """Print detailed analysis report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE DECIMATION ANALYSIS REPORT")
        print("="*80)
        
        for name, analysis in self.analysis_results.items():
            print(f"\n{'-'*60}")
            print(f"DATASET: {name.upper()}")
            print(f"{'-'*60}")
            
            data = self.original_data[name]
            print(f"Original shape: {data.shape}")
            print(f"Total samples: {data.shape[0]:,}")
            
            # Temporal characteristics
            print(f"\nTemporal Analysis:")
            if data.ndim > 1:
                for dim_name, results in analysis['temporal'].items():
                    print(f"  {dim_name}:")
                    print(f"    Correlation length: {results['correlation_length']}")
                    print(f"    Max frequency: {results['max_frequency']:.6f}")
                    print(f"    Approximate entropy: {results['approximate_entropy']:.4f}")
            else:
                results = analysis['temporal'][name]
                print(f"  Correlation length: {results['correlation_length']}")
                print(f"  Max frequency: {results['max_frequency']:.6f}")
                print(f"  Approximate entropy: {results['approximate_entropy']:.4f}")
            
            # Recommendations
            print(f"\nDecimation Recommendations:")
            factors = self.recommended_factors[name]
            print(f"  Spectral-based: {factors['individual_factors']['spectral']}")
            print(f"  Information-based: {factors['individual_factors']['information']}")
            print(f"  Correlation-based: {factors['individual_factors']['correlation']}")
            print(f"\nFinal Recommendations:")
            print(f"  Conservative: {factors['conservative']} ({data.shape[0]//factors['conservative']:,} samples)")
            print(f"  Balanced: {factors['balanced']} ({data.shape[0]//factors['balanced']:,} samples)")
            print(f"  Aggressive: {factors['aggressive']} ({data.shape[0]//factors['aggressive']:,} samples)")

def main():
    """Main analysis and decimation workflow."""
    decimator = DataDecimator()
    
    # Perform comprehensive analysis
    decimator.analyze_all_datasets()
    
    # Generate visualizations
    decimator.visualize_analysis()
    
    # Print detailed report
    decimator.print_comprehensive_report()
    
    # Generate decimated files for all strategies
    strategies = ['conservative', 'balanced', 'aggressive']
    
    print(f"\n{'='*80}")
    print("GENERATING DECIMATED FILES")
    print("="*80)
    
    for strategy in strategies:
        decimator.generate_decimated_files(strategy)
    
    print(f"\nDecimation analysis complete!")
    print(f"Generated files:")
    print(f"  - Analysis visualization: decimation_analysis.png")
    print(f"  - Decimated data files: *_decimated_*.csv")
    print(f"  - Summary reports: decimation_summary_*.csv")

if __name__ == "__main__":
    main()
