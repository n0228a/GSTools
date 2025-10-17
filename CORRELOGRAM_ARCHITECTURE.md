# Correlogram Architecture for GSTools Cokriging

## Overview

This document describes the new Correlogram architecture implemented for collocated cokriging in GSTools. This architecture makes it easy to add new cross-covariance models (MM2, etc.) without modifying existing kriging classes.

## Architecture Design

### Class Hierarchy

```
Correlogram (ABC)                    # Abstract base class
├── MarkovModel1                     # MM1 implementation (current)
└── MarkovModel2                     # MM2 implementation (future)
    
CollocatedCokriging (Krige)         # Base cokriging class
├── SimpleCollocated                 # SCCK algorithm
└── IntrinsicCollocated             # ICCK algorithm
```

### Key Concepts

1. **Separation of Concerns**: 
   - `Correlogram` classes define cross-covariance structure
   - `CollocatedCokriging` classes implement kriging algorithms

2. **Correlogram Parameters**:
   - `primary_model`: CovModel for primary variable
   - `cross_corr`: Cross-correlation at zero lag
   - `secondary_var`: Variance of secondary variable
   - `primary_mean`: Mean of primary variable
   - `secondary_mean`: Mean of secondary variable

3. **Abstract Methods**:
   - `compute_covariances()`: Returns (C_Z0, C_Y0, C_YZ0)
   - `cross_covariance(h)`: Computes C_YZ(h) at distance h

## Usage Examples

### Basic Usage with MarkovModel1

```python
import gstools as gs
import numpy as np

# Define primary variable model
model = gs.Gaussian(dim=1, var=0.5, len_scale=2.0)

# Create MarkovModel1 correlogram
correlogram = gs.MarkovModel1(
    primary_model=model,
    cross_corr=0.8,
    secondary_var=1.5,
    primary_mean=1.0,
    secondary_mean=0.5
)

# Simple Collocated Cokriging
cond_pos = [0.5, 2.1, 3.8]
cond_val = np.array([0.8, 1.2, 1.8])
scck = gs.SimpleCollocated(correlogram, cond_pos, cond_val)

# Interpolate
gridx = np.linspace(0.0, 5.0, 51)
secondary_data = np.ones(51) * 0.5
field = scck(gridx, secondary_data=secondary_data)
```

### Intrinsic Collocated Cokriging

```python
# Requires secondary data at primary locations
sec_at_primary = np.array([0.4, 0.6, 0.7])

icck = gs.IntrinsicCollocated(
    correlogram,
    cond_pos=cond_pos,
    cond_val=cond_val,
    secondary_cond_pos=cond_pos,
    secondary_cond_val=sec_at_primary
)

field_icck = icck(gridx, secondary_data=secondary_data)
```

### Backward Compatibility (Deprecated)

```python
# Old API still works with deprecation warning
scck = gs.SimpleCollocated.from_parameters(
    model, cond_pos, cond_val,
    cross_corr=0.8,
    secondary_var=1.5,
    mean=1.0,
    secondary_mean=0.5
)
```

## Adding New Correlogram Models

### Example: Implementing MarkovModel2

MarkovModel2 uses the secondary variable's spatial structure instead of the primary:

**Formula**: `C_YZ(h) = (C_YZ(0) / C_Y(0)) * C_Y(h)`

**Implementation** (in `src/gstools/cokriging/correlogram/markov.py`):

```python
class MarkovModel2(Correlogram):
    """
    Markov Model II correlogram for collocated cokriging.
    
    Uses the secondary variable's spatial structure for cross-covariance.
    This is useful when the secondary variable has a more stable or 
    better-defined spatial structure than the primary variable.
    """
    
    def __init__(
        self,
        primary_model,
        secondary_model,  # NEW: needs secondary model
        cross_corr,
        secondary_var,
        primary_mean=0.0,
        secondary_mean=0.0,
    ):
        super().__init__(
            primary_model, cross_corr, secondary_var,
            primary_mean, secondary_mean
        )
        self.secondary_model = secondary_model
    
    def compute_covariances(self):
        """Compute covariances at zero lag (same as MM1)."""
        C_Z0 = self.primary_model.sill
        C_Y0 = self.secondary_var
        C_YZ0 = self.cross_corr * np.sqrt(C_Z0 * C_Y0)
        return C_Z0, C_Y0, C_YZ0
    
    def cross_covariance(self, h):
        """
        Compute cross-covariance using MM2 formula.
        
        MM2: C_YZ(h) = (C_YZ(0) / C_Y(0)) * C_Y(h)
        """
        C_Z0, C_Y0, C_YZ0 = self.compute_covariances()
        
        if C_Y0 < 1e-15:
            return np.zeros_like(h) if isinstance(h, np.ndarray) else 0.0
        
        # MM2 formula: uses SECONDARY covariance structure
        k = C_YZ0 / C_Y0
        C_Y_h = self.secondary_model.covariance(h)
        return k * C_Y_h
```

**Usage**:

```python
# Define both primary and secondary models
primary_model = gs.Gaussian(dim=1, var=0.5, len_scale=2.0)
secondary_model = gs.Exponential(dim=1, var=1.5, len_scale=3.0)

# Create MM2 correlogram
mm2 = gs.MarkovModel2(
    primary_model=primary_model,
    secondary_model=secondary_model,
    cross_corr=0.8,
    secondary_var=1.5,
    primary_mean=1.0,
    secondary_mean=0.5
)

# Use with existing kriging classes (no changes needed!)
scck = gs.SimpleCollocated(mm2, cond_pos, cond_val)
```

## Benefits of This Architecture

1. **Extensibility**: Add new correlogram models without touching kriging code
2. **Clarity**: Explicit about which cross-covariance model is being used
3. **Testability**: Correlogram classes can be unit-tested independently
4. **Maintainability**: Clean separation between modeling and interpolation
5. **Future-Proof**: Ready for MM2, Linear Model of Coregionalization, etc.

## File Structure

```
src/gstools/cokriging/
├── correlogram/
│   ├── __init__.py          # Exports Correlogram, MarkovModel1
│   ├── base.py              # Correlogram ABC
│   └── markov.py            # MarkovModel1, (future: MarkovModel2)
├── base.py                  # CollocatedCokriging (refactored)
├── methods.py               # SimpleCollocated, IntrinsicCollocated
└── __init__.py              # Exports all public classes
```

## Testing

Run the correlogram test suite:

```bash
pytest tests/test_correlogram.py -v
```

Tests include:
- MarkovModel1 initialization and validation
- Covariance computation correctness
- Numerical equivalence between old and new API
- Both SimpleCollocated and IntrinsicCollocated

## Migration Guide

### For Users

**Old Code**:
```python
scck = gs.SimpleCollocated(
    model, cond_pos, cond_val,
    cross_corr=0.8, secondary_var=1.5,
    mean=1.0, secondary_mean=0.5
)
```

**New Code** (recommended):
```python
correlogram = gs.MarkovModel1(
    primary_model=model,
    cross_corr=0.8,
    secondary_var=1.5,
    primary_mean=1.0,
    secondary_mean=0.5
)
scck = gs.SimpleCollocated(correlogram, cond_pos, cond_val)
```

**Transitional** (if immediate migration not possible):
```python
scck = gs.SimpleCollocated.from_parameters(
    model, cond_pos, cond_val,
    cross_corr=0.8, secondary_var=1.5,
    mean=1.0, secondary_mean=0.5
)
# Warning: DeprecationWarning will be shown
```

### For Developers

To add a new correlogram model:

1. Create class inheriting from `Correlogram`
2. Implement `compute_covariances()` and `cross_covariance(h)`
3. Add validation in `_validate()` if needed
4. Export from `correlogram/__init__.py`
5. Add to top-level `gstools.__init__.py`
6. Write tests in `tests/test_correlogram.py`

**No changes needed** to `SimpleCollocated` or `IntrinsicCollocated`!

## Future Enhancements

Potential correlogram models to add:

- **MarkovModel2**: Uses secondary variable's spatial structure
- **LinearModelCoregionalization**: Full multivariate model
- **IntrinsicCorrelation**: For intrinsically correlated data
- **HeterotopicModel**: For different sampling locations

All can be added by creating new `Correlogram` subclasses!

## References

- Samson, M., & Deutsch, C. V. (2020). Collocated Cokriging. In J. L. Deutsch (Ed.), Geostatistics Lessons. http://geostatisticslessons.com/lessons/collocatedcokriging
- Wackernagel, H. (2003). Multivariate Geostatistics. Springer, Berlin.

---

**Branch**: `feature/correlogram-architecture`  
**Date**: 2025-10-17  
**Status**: ✅ Complete and tested
