# wedowind_docto

# Links de Interes
# https://github.com/deel-ai/puncc/blob/main/docs/puncc_intro.ipynb
Proyecto WeDoWind

General description of wind turbine: The ETH owned wind turbine is Aventa AV-7, manufactured by Aventa AG in Switzerland and was commissioned in December 2002. The turbine is operated via a belt-driven generator and a frequency converter with a variable speed drive. The rated power of the Aventa AV-7 is 7 kW, beginning production at a wind speed of 2 m/s and having a cut-off speed of 14 m/s. The rotor diameter is 12.8 m with 3 rotor blades, and a hub height is 18m. The maximum rotational speed of the turbine is 63 rpm. The tower is a tubular steel-reinforced concrete structure, supported on concrete foundation, while the blades are made of glassfiber with a tubular steel main-spar. The turbine is regulated via a variable-speed and variable pitch control system.

Location of site: The wind turbine is located in Taggenberg, about 5 km from the city centre of Winterthur, Switzerland. This site is easily accessible by public transport and on foot with direct road access right next to the turbine. This prime location reduces the cost of site visits and allows for frequent personal monitoring of the site when test equipment is installed. The coordinates of the site are: 47°31'12.2"N 8°40'55.7"E.

Control and measurement systems and signals: The turbine is regulated via a variable-speed and collective variable pitch control system.

SHM Motivation: Designed and commissioned in 2002, the Aventa wind turbine in Winterthur is soon reaching its end of design lifetime. In order to assess the various techniques of predicting the remaining useful lifetime, a Structural Health Monitoring (SHM) campaign was implemented by ETH Zurich. The monitoring campaign started in 2020, and is still ongoing. In addition, the setup is used as a research platform on topics such as system identification, operational modal analysis, faults/damage detection and classification. We analyze the influence of operational and environmental conditions on the modal parameters and to further infer Performance Indicators (PIs) for assessing structural behavior in terms of deterioration processes.

Data Description: The tower and nacelle have been instrumented with 11 accelerometers distributed along the length of the tower, nacelle main frame, main bearing and generator. Two full bridge strain gauges are installed on the concrete tower based measuring fore-aft and side-side strain (and can be converted to bending moments) – all acceleration and strain signals sampled at 200Hz. Temperature and humidity are measured at the tower base – 1Hz data. In additional we are collecting operational performance data (SCADA), namely: wind speed, nacelle yaw orientation, rotor RPM, power output and turbine status – SCADA signals are sampled at 10Hz. See appendix for further details of the sensors layout.

The measurements/instrumentation setup, type and layout is provided in the pdf files.

The data: the data is provided in zip files corresponding to four use-cases as follows:

    Normal operation data for system identification
    Aerodynamic imbalance on one blade
    Rotor icing event
    Failure of the flexible coupling of the linear drive of the collective pitch system

The data for each of the four uses-cases is organized in zip files. The content of each zip file is as follows:

    Time-series data in HDF5 format
    Metadata:
        Turbine specification (Aventa-AV-7.json and Aventa-AV-7.yaml)
        Sensor specification (Aventa_sensors.json )
        Unstructured description of the Aventa Turbine and the installed sensors (Aventa_Sensors_Specs.xlsx)
    Semantic artifacts:
        WindIO Wind Turbine YAML schema describing turbine specifications (IEAontology_schema.yaml)
        Sensor specification JSON schema (sensors_schema.json)
    Media: Pictures of leading edge roughness and a clip of wind turbine operation
    Code: Jupyter notebook containing example code to load metadata from JSON and data from HDF5 files (example.ipynb)

Additional data is available upon request, please contact:

    Prof. Dr. Eleni Chatzi (chatzi@ibk.baug.ethz.ch)
    Dr. Imad Abdallah (ai@rtdt.ai , abdallah@ibk.baug.ethz.ch)

For further details or questions, please contact:

Prof. Dr. Eleni Chatzi
Chair of Structural Mechanics & Monitoring

ETH Zürich
http://www.chatzi.ibk.ethz.ch/

