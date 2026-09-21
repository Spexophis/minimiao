import math


# ============================================================
# USER PARAMETERS
# ============================================================

# Measured optical power at the objective back pupil
power_mW = 0.025

# Laser wavelength
wavelength_nm = 488.0

# Objective numerical aperture
NA = 1.3

# Refractive index of the immersion/sample medium
refractive_index = 1.33

# Objective transmission
objective_transmission = 0.90       # 90%

# Additional optical transmission/losses after the
# back-pupil power measurement
additional_transmission = 1.00      # 100%

# How much of the objective back pupil is illuminated
# 1.0 = 100%, 0.5 = 50%, etc.
pupil_fill_factor = 1.0

# Calculation method:
# "gaussian" -> peak intensity of Gaussian beam
# "airy"     -> average intensity inside first Airy disk
calculation_method = "airy"


# ============================================================
# CALCULATIONS
# ============================================================

# Unit conversions
power_W = power_mW * 1e-3
wavelength_m = wavelength_nm * 1e-9

# Effective NA based on pupil illumination
effective_NA = NA * pupil_fill_factor

# Power actually reaching the focal region
focal_power_W = (
    power_W
    * objective_transmission
    * additional_transmission
)


# ------------------------------------------------------------
# GAUSSIAN BEAM
# ------------------------------------------------------------

if calculation_method.lower() == "gaussian":

    # Approximate diffraction-limited Gaussian beam waist
    #
    # w0 = lambda / (pi * NA)
    #
    # w0 is the 1/e^2 intensity radius.

    waist_m = wavelength_m / (
        math.pi * effective_NA
    )

    # Peak intensity:
    #
    # I_peak = 2P / (pi*w0^2)

    power_density_W_m2 = (
        2 * focal_power_W
        / (math.pi * waist_m**2)
    )

    focal_radius_m = waist_m
    focal_diameter_m = 2 * waist_m

    calculation_description = (
        "Gaussian beam peak intensity"
    )


# ------------------------------------------------------------
# AIRY DISK
# ------------------------------------------------------------

elif calculation_method.lower() == "airy":

    # Radius to first Airy minimum
    #
    # r = 0.61 * lambda / NA

    airy_radius_m = (
        0.61
        * wavelength_m
        / effective_NA
    )

    airy_area_m2 = math.pi * airy_radius_m**2

    # Average power density inside the first Airy disk

    power_density_W_m2 = (
        focal_power_W
        / airy_area_m2
    )

    focal_radius_m = airy_radius_m
    focal_diameter_m = 2 * airy_radius_m

    calculation_description = (
        "Airy disk average intensity"
    )


else:
    raise ValueError(
        "calculation_method must be 'gaussian' or 'airy'"
    )


# ============================================================
# UNIT CONVERSIONS
# ============================================================

power_density_W_cm2 = power_density_W_m2 / 1e4
power_density_W_mm2 = power_density_W_m2 / 1e6
power_density_MW_cm2 = power_density_W_cm2 / 1e6

focal_radius_nm = focal_radius_m * 1e9
focal_radius_um = focal_radius_m * 1e6

focal_diameter_nm = focal_diameter_m * 1e9
focal_diameter_um = focal_diameter_m * 1e6


# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("FOCAL POWER DENSITY")
print("=" * 60)

print("\nINPUT PARAMETERS")
print("-" * 60)

print(f"Power at back pupil:       {power_mW:.6g} mW")
print(f"Wavelength:                {wavelength_nm:.6g} nm")
print(f"Objective NA:              {NA:.6g}")
print(f"Refractive index:          {refractive_index:.6g}")
print(f"Objective transmission:    {objective_transmission * 100:.3f} %")
print(f"Additional transmission:   {additional_transmission * 100:.3f} %")
print(f"Pupil fill factor:         {pupil_fill_factor * 100:.3f} %")

print("\nCALCULATED PARAMETERS")
print("-" * 60)

print(f"Effective NA:              {effective_NA:.6g}")
print(f"Power at focus:            {focal_power_W * 1e3:.6g} mW")

print("\nFOCAL SPOT")
print("-" * 60)

print(f"Focal radius:               {focal_radius_nm:.6g} nm")
print(f"Focal radius:               {focal_radius_um:.6g} µm")
print(f"Focal diameter:             {focal_diameter_nm:.6g} nm")
print(f"Focal diameter:             {focal_diameter_um:.6g} µm")

print("\nPOWER DENSITY")
print("-" * 60)

print(f"W/m²:                       {power_density_W_m2:.6e}")
print(f"W/cm²:                      {power_density_W_cm2:.6e}")
print(f"W/mm²:                      {power_density_W_mm2:.6e}")
print(f"MW/cm²:                     {power_density_MW_cm2:.6e}")

print("\nCALCULATION")
print("-" * 60)
print(calculation_description)

print("=" * 60)
