# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# Basic analysis tutorial
o2physics_add_dpl_workflow(my-example-task
                  SOURCES myExampleTask.cxx
                  PUBLIC_LINK_LIBRARIES O2Physics::AnalysisCore
                  COMPONENT_NAME AnalysisTutorial)

# This will generate a file named "o2-<component_name>-<subcomponent_or_subdirectory_group>-<taskname>", thus:
# o2-analysistutorial-mm-my-example-task

# Second example task -- Testing by myself:
o2physics_add_dpl_workflow(my-example-task-2
                  SOURCES myExampleTask2.cxx
                  PUBLIC_LINK_LIBRARIES O2Physics::AnalysisCore
                  COMPONENT_NAME AnalysisTutorial)