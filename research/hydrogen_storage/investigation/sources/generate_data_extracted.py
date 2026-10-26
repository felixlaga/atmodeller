import logging

import numpy as np
import optimistix as optx

from atmodeller import (
    EquilibriumModel,
    Planet,
    SolverParameters,
    ChemicalSpecies,
    SpeciesNetwork,
    debug_logger,
)
from atmodeller.eos import get_eos_models
from atmodeller.solubility import get_solubility_models

logger = debug_logger()
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd


# Palme and O'Neill (2014) Treatise on Geochemistry - Table 3

SiO2_mantlemasspercent_palme14: float = 45.4
total_mantlemasspercent_palme14: float = 98.41
core_mass_fraction: float = 0.327  # typical values used are between 0.325 - 0.33
Si_massfraction_palme14: float = round(
    SiO2_mantlemasspercent_palme14
    / total_mantlemasspercent_palme14
    * (1 - core_mass_fraction)
    / (28.0855 + 2 * 15.999)
    * 28.0855,
    3,
)

O_massfraction_palme14: float = Si_massfraction_palme14 / 28.0855 * 2 * 15.999


# Lodders et al. (2009) Springer Book Chapter - Table 8 (in wt%)

H_masspercent_lodders09: float = 73.9
He_masspercent_lodders09: float = 24.69
C_masspercent_lodders09: float = 0.22
N_masspercent_lodders09: float = 0.07
O_masspercent_lodders09: float = 0.63
Si_masspercent_lodders09: float = 0.07


# Lodders et al. (2009) Springer Book Chapter - Table 6 (log-normalalized abundances relative to H)

H_logN: float = 12
He_logN: float = 10.93
C_logN: float = 8.39
N_logN: float = 7.86
O_logN: float = 8.73
Si_logN: float = 7.53

# Mass and radius of TOI-421b
planet_mass = 6.7 * 5.972e24  # kg
MEB_radius = 1.65 * 6371000  # metre
atm_radius = 2.64 * 6371000  # metre
# MEB radius = 1.65 Earth radii in metre for 6.7 Earth masses (for atm_radius = 2.64 Earth radii)
# M-R relation from Hakim et al. (2018) Icarus

# Temperature of TOI-421b
MEB_temperature = 3000  # K
atm_temperature = 920  # K

number_of_realisations = 50

# For envelope composition setup
hmps = np.logspace(-1, 0, num=number_of_realisations)  # wt% H
h_kgs = hmps / 100 * planet_mass  # kg

si_kg_magma: float = Si_massfraction_palme14 * planet_mass
o_kg_magma: float = O_massfraction_palme14 * planet_mass

# Lodders et al. (2009) Springer book chapter Table 8
si_kgs_solar = h_kgs * Si_masspercent_lodders09 / H_masspercent_lodders09
o_kgs_solar = h_kgs * O_masspercent_lodders09 / H_masspercent_lodders09
c_kgs_solar = h_kgs * C_masspercent_lodders09 / H_masspercent_lodders09
n_kgs_solar = h_kgs * N_masspercent_lodders09 / H_masspercent_lodders09
he_kgs_solar = h_kgs * He_masspercent_lodders09 / H_masspercent_lodders09

# For atmosphere composition setup
atm_scalings = np.logspace(-10, 0, num=number_of_realisations)

h_kg = 1 / 100 * planet_mass  # kg for 1 wt% H

# Lodders et al. (2009) Springer book chapter Table 8
si_kg_solar = h_kg * Si_masspercent_lodders09 / H_masspercent_lodders09
o_kg_solar = h_kg * O_masspercent_lodders09 / H_masspercent_lodders09
c_kg_solar = h_kg * C_masspercent_lodders09 / H_masspercent_lodders09
n_kg_solar = h_kg * N_masspercent_lodders09 / H_masspercent_lodders09
he_kg_solar = h_kg * He_masspercent_lodders09 / H_masspercent_lodders09

eos_models = get_eos_models()
sol_models = get_solubility_models()

H2O_g = ChemicalSpecies.create_gas("H2O")
H2O_gs = ChemicalSpecies.create_gas("H2O", solubility=sol_models["H2O_peridotite_sossi23"])
H2O_rgs = ChemicalSpecies.create_gas(
    "H2O",
    activity=eos_models["H2O_cork_holland98"],
    solubility=sol_models["H2O_peridotite_sossi23"],
)

H2_g = ChemicalSpecies.create_gas("H2")
H2_gs = ChemicalSpecies.create_gas("H2", solubility=sol_models["H2_basalt_hirschmann12"])
H2_rgs = ChemicalSpecies.create_gas(
    "H2", activity=eos_models["H2_chabrier21"], solubility=sol_models["H2_basalt_hirschmann12"]
)

O2_g = ChemicalSpecies.create_gas("O2")
O2_rg = ChemicalSpecies.create_gas("O2", activity=eos_models["O2_cs_shi92"])

OSi_g = ChemicalSpecies.create_gas("OSi")
OSi_rg = ChemicalSpecies.create_gas("OSi", activity=eos_models["OSi_rk49_connolly16"])

H4Si_g = ChemicalSpecies.create_gas("H4Si")
H4Si_rg = ChemicalSpecies.create_gas("H4Si", activity=eos_models["H4Si_wang18"])

O2Si_cd = ChemicalSpecies.create_condensed("O2Si", state="cd")

C_cd = ChemicalSpecies.create_condensed("C", state="cd")
CSi_cd = ChemicalSpecies.create_condensed("CSi", state="cd")
Si_cd = ChemicalSpecies.create_condensed("Si", state="cd")
N4Si3_cd = ChemicalSpecies.create_condensed("N4Si3", state="cd")

CO2_g = ChemicalSpecies.create_gas("CO2")
CO2_gs = ChemicalSpecies.create_gas("CO2", solubility=sol_models["CO2_basalt_dixon95"])
CO2_rgs = ChemicalSpecies.create_gas(
    "CO2", solubility=sol_models["CO2_basalt_dixon95"], activity=eos_models["CO2_cs_shi92"]
)

CO_g = ChemicalSpecies.create_gas("CO")
CO_gs = ChemicalSpecies.create_gas("CO", solubility=sol_models["CO_basalt_yoshioka19"])
CO_rgs = ChemicalSpecies.create_gas(
    "CO", solubility=sol_models["CO_basalt_yoshioka19"], activity=eos_models["CO_cs_shi92"]
)

CH4_g = ChemicalSpecies.create_gas("CH4")
CH4_gs = ChemicalSpecies.create_gas("CH4", solubility=sol_models["CH4_basalt_ardia13"])
CH4_rgs = ChemicalSpecies.create_gas(
    "CH4", solubility=sol_models["CH4_basalt_ardia13"], activity=eos_models["CH4_cs_shi92"]
)

N2_g = ChemicalSpecies.create_gas("N2")
N2_gs = ChemicalSpecies.create_gas("N2", solubility=sol_models["N2_basalt_libourel03"])
N2_rgs = ChemicalSpecies.create_gas(
    "N2", solubility=sol_models["N2_basalt_libourel03"], activity=eos_models["N2_cs_saxena87"]
)

NH3_g = ChemicalSpecies.create_gas("H3N")
NH3_rg = ChemicalSpecies.create_gas("H3N", activity=eos_models["H3N_rk49_reid87"])

He_g = ChemicalSpecies.create_gas("He")
He_gs = ChemicalSpecies.create_gas("He", solubility=sol_models["He_basalt_jambon86"])
He_rgs = ChemicalSpecies.create_gas(
    "He", solubility=sol_models["He_basalt_jambon86"], activity=eos_models["He_chabrier21"]
)


species_HOSi_magma_nosol_ideal = SpeciesNetwork((H2O_g, H2_g, O2_g, OSi_g, H4Si_g, O2Si_cd))
species_HOSi_magma_sol_ideal = SpeciesNetwork((H2O_gs, H2_gs, O2_g, OSi_g, H4Si_g, O2Si_cd))
species_HOSi_magma_sol_real = SpeciesNetwork((H2O_rgs, H2_rgs, O2_rg, OSi_rg, H4Si_rg, O2Si_cd))


species_HHeCNOSi_magma_nosol_ideal = SpeciesNetwork(
    (H2O_g, H2_g, O2_g, OSi_g, H4Si_g, O2Si_cd, CO2_g, CO_g, CH4_g, N2_g, NH3_g, He_g)
)
species_HHeCNOSi_magma_sol_ideal = SpeciesNetwork(
    (H2O_gs, H2_gs, O2_g, OSi_g, H4Si_g, O2Si_cd, CO2_gs, CO_gs, CH4_gs, N2_gs, NH3_g, He_gs)
)
species_HHeCNOSi_magma_sol_real = SpeciesNetwork(
    (
        H2O_rgs,
        H2_rgs,
        O2_rg,
        OSi_rg,
        H4Si_rg,
        O2Si_cd,
        CO2_rgs,
        CO_rgs,
        CH4_rgs,
        N2_rgs,
        NH3_rg,
        He_rgs,
    )
)

species_HHeCNOSi_atmosphere = SpeciesNetwork(
    (H2O_g, H2_g, O2_g, OSi_g, H4Si_g, O2Si_cd, C_cd, CSi_cd, Si_cd, N4Si3_cd,
     CO2_g, CO_g, CH4_g, N2_g, NH3_g, He_g)
)

# Update accreted gas metallicity below
init_metallicity = 1  # 1x solar

# Update mantle melt fraction below
mantle_melt_fraction = 1

planet = Planet(
    temperature=MEB_temperature,
    planet_mass=planet_mass,
    surface_radius=MEB_radius,
    mantle_melt_fraction=mantle_melt_fraction,
)

mass_constraints = {
    "Si": si_kg_magma * mantle_melt_fraction + si_kgs_solar * init_metallicity,
    "O": o_kg_magma * mantle_melt_fraction + o_kgs_solar * init_metallicity,
    "H": h_kgs,
}

# Magma - No Solubility - Ideal Gas
model_magma_nosol_ideal = EquilibriumModel(species_HOSi_magma_nosol_ideal)
model_magma_nosol_ideal.solve(
    state=planet,
    mass_constraints=mass_constraints,
)
output_magma_nosol_ideal = model_magma_nosol_ideal.output
output_magma_nosol_ideal.quick_look()
output_magma_nosol_ideal.to_excel(f"HOSi_magma_nosol_ideal_{init_metallicity}xsolar")

# Magma - Solubility - Ideal Gas
model_magma_sol_ideal = EquilibriumModel(species_HOSi_magma_sol_ideal)
initial_log_number_moles = output_magma_nosol_ideal.log_number_moles
model_magma_sol_ideal.solve(
    state=planet,
    mass_constraints=mass_constraints,
    initial_log_number_moles=initial_log_number_moles,
)
output_magma_sol_ideal = model_magma_sol_ideal.output
output_magma_sol_ideal.quick_look()
output_magma_sol_ideal.to_excel(f"HOSi_magma_sol_ideal_{init_metallicity}xsolar")

# Magma - Solubility - Real Gas
model_magma_sol_real = EquilibriumModel(species_HOSi_magma_sol_real)
initial_log_number_moles = output_magma_sol_ideal.log_number_moles
model_magma_sol_real.solve(
    state=planet,
    mass_constraints=mass_constraints,
    initial_log_number_moles=initial_log_number_moles,
)
output_magma_sol_real = model_magma_sol_real.output
output_magma_sol_real.quick_look()
output_magma_sol_real.to_excel(f"HOSi_magma_sol_real_{init_metallicity}xsolar")

mantle_melt_fractions = [1, 0.3, 0.1, 0.03, 0.01]  # mantle melt fraction
init_metallicitys = [1, 3, 10, 30, 100]  # metallicity in x solar units

# Initialize EquilibriumModel objects 
model_magma_nosol_ideal = EquilibriumModel(species_HHeCNOSi_magma_nosol_ideal)
model_magma_sol_ideal = EquilibriumModel(species_HHeCNOSi_magma_sol_ideal)
model_magma_sol_real = EquilibriumModel(species_HHeCNOSi_magma_sol_real)

for mantle_melt_fraction in mantle_melt_fractions:
    planet = Planet(
        temperature=MEB_temperature,
        planet_mass=planet_mass,
        mantle_melt_fraction=mantle_melt_fraction,
        surface_radius=MEB_radius,
    )

    for init_metallicity in init_metallicitys:
        mass_constraints = {
            "H": h_kgs,
            "He": he_kgs_solar,
            "C": init_metallicity * c_kgs_solar,
            "N": init_metallicity * n_kgs_solar,
            "Si": init_metallicity * si_kgs_solar + mantle_melt_fraction * si_kg_magma,
            "O": init_metallicity * o_kgs_solar + mantle_melt_fraction * o_kg_magma,
        }

        # Magma - No solubility - Ideal Gas
        model_magma_nosol_ideal.solve(
            state=planet,
            mass_constraints=mass_constraints,
        )
        output_magma_nosol_ideal = model_magma_nosol_ideal.output
        # output_magma_nosol_ideal.quick_look()
        output_magma_nosol_ideal.to_excel(
            f"HHeCNOSi_magma_nosol_ideal_{init_metallicity}xsolar_melt{round(mantle_melt_fraction * 100)}"
        )

        # Magma - Solubility - Ideal Gas
        initial_log_number_moles = output_magma_nosol_ideal.log_number_moles
        model_magma_sol_ideal.solve(
            state=planet,
            mass_constraints=mass_constraints,
            initial_log_number_moles=initial_log_number_moles,
        )
        output_magma_sol_ideal = model_magma_sol_ideal.output
        # output_magma_sol_ideal.quick_look()
        output_magma_sol_ideal.to_excel(
            f"HHeCNOSi_magma_sol_ideal_{init_metallicity}xsolar_melt{round(mantle_melt_fraction * 100)}"
        )

        # Magma - Solubility - Real Gas
        initial_log_number_moles = output_magma_sol_ideal.log_number_moles
        model_magma_sol_real.solve(
            state=planet,
            mass_constraints=mass_constraints,
            initial_log_number_moles=initial_log_number_moles,
        )
        output_magma_sol_real = model_magma_sol_real.output
        output_magma_sol_real.quick_look()
        output_magma_sol_real.to_excel(
            f"HHeCNOSi_magma_sol_real_{init_metallicity}xsolar_melt{round(mantle_melt_fraction * 100)}"
        )

mantle_melt_fractions = [1, 0.3, 0.1, 0.03, 0.01]  # mantle melt fraction
init_metallicitys = [1, 3, 10, 30, 100]  # metallicity in x solar units

planet = Planet(
    temperature=atm_temperature,
    planet_mass=planet_mass,
    mantle_melt_fraction=0, # atmosphere has no magma 
    surface_radius=atm_radius,
)

model_atm_magma_sol_real = EquilibriumModel(species_HHeCNOSi_atmosphere)

for mantle_melt_fraction in mantle_melt_fractions:
    for init_metallicity in init_metallicitys:

            # Magma - Solubility - Real Gas
            filename = f"HHeCNOSi_magma_sol_real_{init_metallicity}xsolar_melt{round(mantle_melt_fraction * 100)}.xlsx"

            H_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_H")[
                "gas_mass"
            ][number_of_realisations-1]
            He_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_He")[
                "gas_mass"
            ][number_of_realisations-1]
            C_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_C")[
                "gas_mass"
            ][number_of_realisations-1]
            N_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_N")[
                "gas_mass"
            ][number_of_realisations-1]
            Si_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_Si")[
                "gas_mass"
            ][number_of_realisations-1]
            O_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_O")[
                "gas_mass"
            ][number_of_realisations-1]
            tot_pressure_magma_sol_real = pd.read_excel(filename, sheet_name="state")["pressure"][
                number_of_realisations-1
            ]

            mass_constraints = {
                "H": H_mass_atm_magma_sol_real * atm_scalings,
                "He": He_mass_atm_magma_sol_real * atm_scalings,
                "C": C_mass_atm_magma_sol_real * atm_scalings,
                "N": N_mass_atm_magma_sol_real * atm_scalings,
                "O": O_mass_atm_magma_sol_real * atm_scalings,
                "Si": Si_mass_atm_magma_sol_real * atm_scalings,
            }

            model_atm_magma_sol_real.solve(
                state=planet,
                mass_constraints=mass_constraints,
                solver_parameters=SolverParameters(multistart=20),
            )
            output_atm_magma_sol_real = model_atm_magma_sol_real.output

            output_atm_magma_sol_real.quick_look()

            output_atm_magma_sol_real.to_excel(
                f"HHeCNOSi_atm_magma_sol_real_{init_metallicity}xsolar_melt{round(mantle_melt_fraction * 100)}_1wtH"
            )

for init_metallicity in init_metallicitys:

    mass_constraints = {
        "H": h_kg * atm_scalings,
        "He": he_kg_solar * atm_scalings,
        "C": init_metallicity * c_kg_solar * atm_scalings,
        "N": init_metallicity * n_kg_solar * atm_scalings,
        "O": init_metallicity * o_kg_solar * atm_scalings,
        "Si": init_metallicity * si_kg_solar * atm_scalings,
    }

    model_atm_magma_sol_real = EquilibriumModel(species_HHeCNOSi_atmosphere)
    model_atm_magma_sol_real.solve(
        state=planet,
        mass_constraints=mass_constraints,
        # Usually the multistart solves within 10, but increase just to be sure since the
        # random seed may affect the exact number
        solver_parameters=SolverParameters(multistart=20),
    )
    output_atm_magma_sol_real = model_atm_magma_sol_real.output
    output_atm_magma_sol_real.quick_look()
    output_atm_magma_sol_real.to_excel(
        f"HHeCNOSi_atm_magma_sol_real_{init_metallicity}xsolar_melt0_1wtH"
    )

# setup atmodeller planet object as atmosphere with no magma (melt fraction = 0)

mantle_melt_fraction = 1
atmosphere_temperatures = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]  # K
init_metallicitys = [1, 100]  # metallicity in x solar units

model_atm_magma_sol_real = EquilibriumModel(species_HHeCNOSi_atmosphere)

for atmosphere_temperature in atmosphere_temperatures:
    for init_metallicity in init_metallicitys:

        planet = Planet(
            temperature=atmosphere_temperature,
            planet_mass=planet_mass,
            mantle_melt_fraction=0, # atmosphere has no magma
            surface_radius=atm_radius,
        )

        # Magma - Solubility - Real Gas
        filename = f"HHeCNOSi_magma_sol_real_{init_metallicity}xsolar_melt{round(mantle_melt_fraction * 100)}.xlsx"

        H_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_H")[
            "gas_mass"
        ][number_of_realisations-1]
        He_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_He")[
            "gas_mass"
        ][number_of_realisations-1]
        C_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_C")[
            "gas_mass"
        ][number_of_realisations-1]
        N_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_N")[
            "gas_mass"
        ][number_of_realisations-1]
        Si_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_Si")[
            "gas_mass"
        ][number_of_realisations-1]
        O_mass_atm_magma_sol_real = pd.read_excel(filename, sheet_name="element_O")[
            "gas_mass"
        ][number_of_realisations-1]
        tot_pressure_magma_sol_real = pd.read_excel(filename, sheet_name="state")["pressure"][
            number_of_realisations-1
        ]

        mass_constraints = {
            "H": H_mass_atm_magma_sol_real * atm_scalings,
            "He": He_mass_atm_magma_sol_real * atm_scalings,
            "C": C_mass_atm_magma_sol_real * atm_scalings,
            "N": N_mass_atm_magma_sol_real * atm_scalings,
            "O": O_mass_atm_magma_sol_real * atm_scalings,
            "Si": Si_mass_atm_magma_sol_real * atm_scalings,
        }

        model_atm_magma_sol_real.solve(
            state=planet,
            mass_constraints=mass_constraints,
            solver_parameters=SolverParameters(multistart=20),
        )
        output_atm_magma_sol_real = model_atm_magma_sol_real.output

        output_atm_magma_sol_real.quick_look()

        output_atm_magma_sol_real.to_excel(
            f"HHeCNOSi_atm_magma_sol_real_{init_metallicity}xsolar_melt{round(mantle_melt_fraction * 100)}_1wtH_{atmosphere_temperature}K"
        )

for atmosphere_temperature in atmosphere_temperatures:
    for init_metallicity in init_metallicitys:

        planet = Planet(
            temperature=atmosphere_temperature,
            planet_mass=planet_mass,
            mantle_melt_fraction=0,
            surface_radius=atm_radius,
        )

        mass_constraints = {
            "H": h_kg * atm_scalings,
            "He": he_kg_solar * atm_scalings,
            "C": init_metallicity * c_kg_solar * atm_scalings,
            "N": init_metallicity * n_kg_solar * atm_scalings,
            "O": init_metallicity * o_kg_solar * atm_scalings,
            "Si": init_metallicity * si_kg_solar * atm_scalings,
        }

        model_atm_magma_sol_real = EquilibriumModel(species_HHeCNOSi_atmosphere)
        model_atm_magma_sol_real.solve(
            state=planet,
            mass_constraints=mass_constraints,
            # Usually the multistart solves within 10, but increase just to be sure since the
            # random seed may affect the exact number
            solver_parameters=SolverParameters(multistart=20),
        )
        output_atm_magma_sol_real = model_atm_magma_sol_real.output
        output_atm_magma_sol_real.quick_look()
        output_atm_magma_sol_real.to_excel(
            f"HHeCNOSi_atm_magma_sol_real_{init_metallicity}xsolar_melt0_1wtH_{atmosphere_temperature}K"
        )

error_threshold = 1e-8

mantle_melt_fractions = [1]  # mantle melt fraction
init_metallicitys = [1]  # metallicity in x solar units

for mantle_melt_fraction in mantle_melt_fractions:
    for init_metallicity in init_metallicitys:
        logger.info(
            f"Checking mass balance for mantle melt fraction: {round(100 * mantle_melt_fraction)} % melt and initial metallicity: {init_metallicity}x solar"
        )
        filename = f"HOSi_magma_sol_real_{init_metallicity}xsolar.xlsx"

        Si_total_mass = si_kg_magma * mantle_melt_fraction + si_kgs_solar * init_metallicity
        Si_gas_mass = pd.read_excel(filename, sheet_name="element_Si")[
            "gas_mass"
        ].values
        Si_condensed_mass = pd.read_excel(filename, sheet_name="element_Si")[
            "condensed_mass"
        ].values
        Si_dissolved_mass = pd.read_excel(filename, sheet_name="element_Si")[
            "dissolved_mass"
        ].values
        Si_difference_mass = Si_total_mass - (
            Si_gas_mass + Si_condensed_mass + Si_dissolved_mass
        )
        Si_error_fraction = Si_difference_mass / Si_total_mass

        O_total_mass = o_kg_magma * mantle_melt_fraction + o_kgs_solar * init_metallicity
        O_gas_mass = pd.read_excel(filename, sheet_name="element_O")[
            "gas_mass"
        ].values
        O_condensed_mass = pd.read_excel(filename, sheet_name="element_O")["condensed_mass"].values
        O_dissolved_mass = pd.read_excel(filename, sheet_name="element_O")["dissolved_mass"].values
        O_difference_mass = O_total_mass - (
            O_gas_mass + O_condensed_mass + O_dissolved_mass
        )
        O_error_fraction = O_difference_mass / O_total_mass

        H_total_mass = h_kgs
        H_gas_mass = pd.read_excel(filename, sheet_name="element_H")[
            "gas_mass"
        ].values
        H_condensed_mass = pd.read_excel(filename, sheet_name="element_H")["condensed_mass"].values
        H_dissolved_mass = pd.read_excel(filename, sheet_name="element_H")["dissolved_mass"].values
        H_difference_mass = H_total_mass - (
            H_gas_mass + H_condensed_mass + H_dissolved_mass
        )
        H_error_fraction = H_difference_mass / H_total_mass

        if (
            any(abs(Si_error_fraction) > error_threshold)
            or any(abs(O_error_fraction) > error_threshold)
            or any(abs(H_error_fraction) > error_threshold)
        ):
            logger.warning(
                f"Mass balance error exceeds {error_threshold} for one or more elements."
            )
        error_counter_Si = error_counter_O = error_counter_H = 0
        for i in range(number_of_realisations):
            if abs(Si_error_fraction[i]) > error_threshold:
                error_counter_Si = error_counter_Si + 1
            if abs(O_error_fraction[i]) > error_threshold:
                error_counter_O = error_counter_O + 1
            if abs(H_error_fraction[i]) > error_threshold:
                error_counter_H = error_counter_H + 1

        logger.info(
            f"Number of errors: Si = {error_counter_Si}, O = {error_counter_O}, H = {error_counter_H}"
        )
        logger.info(f"Si Mass Balance Error: {any(abs(Si_error_fraction) > error_threshold)}")
        logger.info(f"O Mass Balance Error: {any(abs(O_error_fraction) > error_threshold)}")
        logger.info(f"H Mass Balance Error: {any(abs(H_error_fraction) > error_threshold)}")

error_threshold = 1e-8

mantle_melt_fractions = [1, 0.3, 0.1, 0.03, 0.01]  # mantle melt fraction
init_metallicitys = [1, 3, 10, 30, 100]  # metallicity in x solar units

for mantle_melt_fraction in mantle_melt_fractions:
    for init_metallicity in init_metallicitys:
        logger.info(
            f"Checking mass balance for mantle melt fraction: {round(100 * mantle_melt_fraction)} % melt and initial metallicity: {init_metallicity}x solar"
        )
        filename = f"HHeCNOSi_magma_sol_real_{init_metallicity}xsolar_melt{round(mantle_melt_fraction * 100)}.xlsx"

        Si_total_mass = si_kg_magma * mantle_melt_fraction + si_kgs_solar * init_metallicity
        Si_gas_mass = pd.read_excel(filename, sheet_name="element_Si")[
            "gas_mass"
        ].values
        Si_condensed_mass = pd.read_excel(filename, sheet_name="element_Si")[
            "condensed_mass"
        ].values
        Si_dissolved_mass = pd.read_excel(filename, sheet_name="element_Si")[
            "dissolved_mass"
        ].values
        Si_difference_mass = Si_total_mass - (
            Si_gas_mass + Si_condensed_mass + Si_dissolved_mass
        )
        Si_error_fraction = Si_difference_mass / Si_total_mass

        O_total_mass = o_kg_magma * mantle_melt_fraction + o_kgs_solar * init_metallicity
        O_gas_mass = pd.read_excel(filename, sheet_name="element_O")[
            "gas_mass"
        ].values
        O_condensed_mass = pd.read_excel(filename, sheet_name="element_O")["condensed_mass"].values
        O_dissolved_mass = pd.read_excel(filename, sheet_name="element_O")["dissolved_mass"].values
        O_difference_mass = O_total_mass - (
            O_gas_mass + O_condensed_mass + O_dissolved_mass
        )
        O_error_fraction = O_difference_mass / O_total_mass

        H_total_mass = h_kgs
        H_gas_mass = pd.read_excel(filename, sheet_name="element_H")[
            "gas_mass"
        ].values
        H_condensed_mass = pd.read_excel(filename, sheet_name="element_H")["condensed_mass"].values
        H_dissolved_mass = pd.read_excel(filename, sheet_name="element_H")["dissolved_mass"].values
        H_difference_mass = H_total_mass - (
            H_gas_mass + H_condensed_mass + H_dissolved_mass
        )
        H_error_fraction = H_difference_mass / H_total_mass

        He_total_mass = he_kgs_solar
        He_gas_mass = pd.read_excel(filename, sheet_name="element_He")[
            "gas_mass"
        ].values
        He_condensed_mass = pd.read_excel(filename, sheet_name="element_He")[
            "condensed_mass"
        ].values
        He_dissolved_mass = pd.read_excel(filename, sheet_name="element_He")[
            "dissolved_mass"
        ].values
        He_difference_mass = He_total_mass - (
            He_gas_mass + He_condensed_mass + He_dissolved_mass
        )
        He_error_fraction = He_difference_mass / He_total_mass

        C_total_mass = c_kgs_solar * init_metallicity
        C_gas_mass = pd.read_excel(filename, sheet_name="element_C")[
            "gas_mass"
        ].values
        C_condensed_mass = pd.read_excel(filename, sheet_name="element_C")["condensed_mass"].values
        C_dissolved_mass = pd.read_excel(filename, sheet_name="element_C")["dissolved_mass"].values
        C_difference_mass = C_total_mass - (
            C_gas_mass + C_condensed_mass + C_dissolved_mass
        )
        C_error_fraction = C_difference_mass / C_total_mass

        N_total_mass = n_kgs_solar * init_metallicity
        N_gas_mass = pd.read_excel(filename, sheet_name="element_N")[
            "gas_mass"
        ].values
        N_condensed_mass = pd.read_excel(filename, sheet_name="element_N")["condensed_mass"].values
        N_dissolved_mass = pd.read_excel(filename, sheet_name="element_N")["dissolved_mass"].values
        N_difference_mass = N_total_mass - (
            N_gas_mass + N_condensed_mass + N_dissolved_mass
        )
        N_error_fraction = N_difference_mass / N_total_mass

        if (
            any(abs(Si_error_fraction) > error_threshold)
            or any(abs(O_error_fraction) > error_threshold)
            or any(abs(H_error_fraction) > error_threshold)
            or any(abs(He_error_fraction) > error_threshold)
            or any(abs(C_error_fraction) > error_threshold)
            or any(abs(N_error_fraction) > error_threshold)
        ):
            logger.warning(
                f"Mass balance error exceeds {error_threshold} for one or more elements."
            )
        error_counter_Si = error_counter_O = error_counter_H = error_counter_He = (
            error_counter_C
        ) = error_counter_N = 0
        for i in range(number_of_realisations):
            if abs(Si_error_fraction[i]) > error_threshold:
                error_counter_Si = error_counter_Si + 1
            if abs(O_error_fraction[i]) > error_threshold:
                error_counter_O = error_counter_O + 1
            if abs(H_error_fraction[i]) > error_threshold:
                error_counter_H = error_counter_H + 1
            if abs(C_error_fraction[i]) > error_threshold:
                error_counter_C = error_counter_C + 1
            if abs(N_error_fraction[i]) > error_threshold:
                error_counter_N = error_counter_N + 1
            if abs(He_error_fraction[i]) > error_threshold:
                error_counter_He = error_counter_He + 1

        logger.info(
            f"Number of errors: Si = {error_counter_Si}, O = {error_counter_O}, H = {error_counter_H}, He = {error_counter_He}, C = {error_counter_C}, N = {error_counter_N}"
        )
        logger.info(f"Si Mass Balance Error: {any(abs(Si_error_fraction) > error_threshold)}")
        logger.info(f"O Mass Balance Error: {any(abs(O_error_fraction) > error_threshold)}")
        logger.info(f"H Mass Balance Error: {any(abs(H_error_fraction) > error_threshold)}")
        logger.info(f"He Mass Balance Error: {any(abs(He_error_fraction) > error_threshold)}")
        logger.info(f"C Mass Balance Error: {any(abs(C_error_fraction) > error_threshold)}")
        logger.info(f"N Mass Balance Error: {any(abs(N_error_fraction) > error_threshold)}")


filename = f"HHeCNOSi_magma_sol_real_1xsolar_melt100.xlsx"

H_moles_atm_1xsolar_1wtH = pd.read_excel(filename, sheet_name="element_H")["gas_number"][number_of_realisations-1]
He_moles_atm_1xsolar_1wtH = pd.read_excel(filename, sheet_name="element_He")["gas_number"][
    number_of_realisations-1
]
C_moles_atm_1xsolar_1wtH = pd.read_excel(filename, sheet_name="element_C")["gas_number"][number_of_realisations-1]
N_moles_atm_1xsolar_1wtH = pd.read_excel(filename, sheet_name="element_N")["gas_number"][number_of_realisations-1]
O_moles_atm_1xsolar_1wtH = pd.read_excel(filename, sheet_name="element_O")["gas_number"][number_of_realisations-1]
Si_moles_atm_1xsolar_1wtH = pd.read_excel(filename, sheet_name="element_Si")["gas_number"][
    number_of_realisations-1
]

Si_metallicity_1xsolar = (
    Si_moles_atm_1xsolar_1wtH / H_moles_atm_1xsolar_1wtH / (10 ** (Si_logN - H_logN))
)
He_metallicity_1xsolar = (
    He_moles_atm_1xsolar_1wtH / H_moles_atm_1xsolar_1wtH / (10 ** (He_logN - H_logN))
)
C_metallicity_1xsolar = (
    C_moles_atm_1xsolar_1wtH / H_moles_atm_1xsolar_1wtH / (10 ** (C_logN - H_logN))
)
N_metallicity_1xsolar = (
    N_moles_atm_1xsolar_1wtH / H_moles_atm_1xsolar_1wtH / (10 ** (N_logN - H_logN))
)
O_metallicity_1xsolar = (
    O_moles_atm_1xsolar_1wtH / H_moles_atm_1xsolar_1wtH / (10 ** (O_logN - H_logN))
)

filename = f"HHeCNOSi_magma_sol_real_100xsolar_melt100.xlsx"

H_moles_atm_100xsolar_1wtH = pd.read_excel(filename, sheet_name="element_H")["gas_number"][
    number_of_realisations-1
]
He_moles_atm_100xsolar_1wtH = pd.read_excel(filename, sheet_name="element_He")["gas_number"][
    number_of_realisations-1
]
C_moles_atm_100xsolar_1wtH = pd.read_excel(filename, sheet_name="element_C")["gas_number"][
    number_of_realisations-1
]
N_moles_atm_100xsolar_1wtH = pd.read_excel(filename, sheet_name="element_N")["gas_number"][
    number_of_realisations-1
]
O_moles_atm_100xsolar_1wtH = pd.read_excel(filename, sheet_name="element_O")["gas_number"][
    number_of_realisations-1
]
Si_moles_atm_100xsolar_1wtH = pd.read_excel(filename, sheet_name="element_Si")["gas_number"][
    number_of_realisations-1
]

Si_metallicity_100xsolar = (
    Si_moles_atm_100xsolar_1wtH / H_moles_atm_100xsolar_1wtH / (10 ** (Si_logN - H_logN))
)
He_metallicity_100xsolar = (
    He_moles_atm_100xsolar_1wtH / H_moles_atm_100xsolar_1wtH / (10 ** (He_logN - H_logN))
)
C_metallicity_100xsolar = (
    C_moles_atm_100xsolar_1wtH / H_moles_atm_100xsolar_1wtH / (10 ** (C_logN - H_logN))
)
N_metallicity_100xsolar = (
    N_moles_atm_100xsolar_1wtH / H_moles_atm_100xsolar_1wtH / (10 ** (N_logN - H_logN))
)
O_metallicity_100xsolar = (
    O_moles_atm_100xsolar_1wtH / H_moles_atm_100xsolar_1wtH / (10 ** (O_logN - H_logN))
)

print("1 wt% H and 1x solar")
print("H", H_logN)
print("He", round(He_logN + np.log10(He_metallicity_1xsolar), 2))
print("C", round(C_logN + np.log10(C_metallicity_1xsolar), 2))
print("N", round(N_logN + np.log10(N_metallicity_1xsolar), 2))
print("O", round(O_logN + np.log10(O_metallicity_1xsolar), 2))
print("Si", round(Si_logN + np.log10(Si_metallicity_1xsolar), 2))
print("")
print("1 wt% H and 100x solar")
print("H", "12")
print("He", round(He_logN + np.log10(He_metallicity_100xsolar), 2))
print("C", round(C_logN + np.log10(C_metallicity_100xsolar), 2))
print("N", round(N_logN + np.log10(N_metallicity_100xsolar), 2))
print("O", round(O_logN + np.log10(O_metallicity_100xsolar), 2))
print("Si", round(Si_logN + np.log10(Si_metallicity_100xsolar), 2))