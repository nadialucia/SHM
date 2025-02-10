import os
from abaqusConstants import *
import logging
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil
import pickle
from abaqusConstants import *

class BeamModel:
    def __init__(self, mdb, config, damage_scenarios):
        self.mdb = mdb
        self.config = config
        self.damage_scenarios = damage_scenarios
        self.model = self.mdb.models['Model-1']
        self.part = None
        self.logger = logging.getLogger('BeamModel')


    def create_part(self):
        self.logger.info("Creating beam part")
        
        # Create sketch
        s = self.model.ConstrainedSketch(name='__profile__', sheetSize=2.0)
        s.Line(point1=(0.0, 0.0), point2=(self.config['geometry']['length'], 0.0))
        
        # Create part and store it
        self.part = self.model.Part(dimensionality=TWO_D_PLANAR, name='Part-1', type=DEFORMABLE_BODY)
        self.part.BaseWire(sketch=s)
        del self.model.sketches['__profile__']

    def partition_beam(self):
        self.logger.info("Partitioning beam")
        if not self.part:
            raise ValueError("Part not created")
            
        length = self.config['geometry']['length']
        n_p = self.config['mesh']['n_p']
        
        for i in range(1, n_p):
            x_coord = i * length/n_p
            edge = self.part.edges.findAt((x_coord - length/(2*n_p), 0.0, 0.0))
            self.part.PartitionEdgeByPoint(
                edge=edge,
                point=self.part.InterestingPoint(edge, MIDDLE)
            )
  
    def create_sections(self):
        logging.info("Creating sections")
        # Create profile
        self.model.RectangularProfile(
            a=self.config['geometry']['width'],
            b=self.config['geometry']['height'],
            name='Profile-1'
        )
        
        # Create sections
        for i in range(self.config['mesh']['n_p']):
            self.model.BeamSection(
                beamSectionOffset=(0.0, 0.0),
                consistentMassMatrix=False,
                integration=DURING_ANALYSIS,
                material=f'Material-{i+1}',
                name=f'Section-{i+1}',
                poissonRatio=0.0,
                profile='Profile-1',
                temperatureVar=LINEAR
            )

    def assign_sections(self):
        logging.info("Assigning sections")
        length = self.config['geometry']['length']
        n_p = self.config['mesh']['n_p']
        
        for i in range(n_p):
            pos = (i + 0.5) * length/n_p
            self.part.SectionAssignment(
                region=Region(
                    edges=self.part.edges.findAt(((pos, 0.0, 0.0), ))
                ),
                sectionName=f'Section-{i+1}'
            )
            
        # Assign beam orientation
        self.part.assignBeamSectionOrientation(
            method=N1_COSINES,
            n1=(0.0, 0.0, -1.0),
            region=Region(edges=self.part.edges)
        )

    def create_assembly(self):
        logging.info("Creating assembly")
        self.assembly = self.model.rootAssembly
        self.assembly.DatumCsysByDefault(CARTESIAN)
        self.assembly.Instance(
            dependent=OFF,
            name='Part-1-1',
            part=self.part
        )

    def create_step(self):
        logging.info("Creating frequency step")
        self.model.FrequencyStep(
            name='Step-1',
            numEigen=self.config['mesh']['n_modes'],
            previous='Initial'
        )

    def create_boundary(self):
        logging.info("Creating boundary conditions")
        self.model.PinnedBC(
            createStepName='Step-1',
            name='BC-1',
            region=Region(
                vertices=self.assembly.instances['Part-1-1'].vertices
            )
        )

    def create_mesh(self):
        logging.info("Creating mesh")
        self.assembly.seedPartInstance(
            deviationFactor=0.1,
            minSizeFactor=0.1,
            regions=(self.assembly.instances['Part-1-1'],),
            size=self.config['geometry']['length']/self.config['mesh']['n_e']
        )
        
        self.assembly.setElementType(
            elemTypes=(ElemType(elemCode=B23, elemLibrary=STANDARD),),
            regions=(self.assembly.instances['Part-1-1'].edges,)
        )
        
        self.assembly.generateMesh()

    def create_mesh(self):
        self.logger.info("Creating mesh")
        
        # Get mesh size from config
        mesh_size = self.config['geometry']['length']/self.config['mesh']['n_e']
        
        # Seed part instance
        self.assembly.seedPartInstance(
            deviationFactor=0.1,
            minSizeFactor=0.1,
            regions=(self.assembly.instances['Part-1-1'], ),
            size=mesh_size
        )
        
        # Get edges for mesh regions
        length = self.config['geometry']['length']
        n_p = self.config['mesh']['n_p']
        edge_positions = [(i + 0.5) * length/n_p for i in range(n_p)]
        
        edges = self.assembly.instances['Part-1-1'].edges
        mesh_regions = edges.findAt(*[((pos, 0.0, 0.0), ) for pos in edge_positions])
        
        # Set element type
        self.assembly.setElementType(
            elemTypes=(ElemType(elemCode=B23, elemLibrary=STANDARD), ),
            regions=(mesh_regions, )
        )
        
        # Generate mesh with specified regions
        self.assembly.generateMesh(
            regions=(self.assembly.instances['Part-1-1'], )
        )

    def create_job(self):
        self.logger.info("Creating and running job")
        
        # Clean up previous job files
        job_name = 'Job-1'
        for ext in ['.lck', '.odb', '.dat', '.msg', '.sta', '.rec']:
            file_path = os.path.join(os.getcwd(), job_name + ext)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    self.logger.warning(f"Could not remove {file_path}")
        
        # Create and run job
        job = self.mdb.Job(name=job_name, model='Model-1')
        job.submit()
        job.waitForCompletion()
        
        # Add delay before accessing ODB
        import time
        time.sleep(20)  # Wait 2 seconds for file system

    def extract_results(self):
        self.logger.info("Extracting results")
        
        # Get current working directory and open ODB
        current_folder = os.getcwd()
        odb_path = os.path.join(current_folder, 'Job-1.odb')
        odb = session.openOdb(odb_path)
        
        try:
            step = odb.steps['Step-1']
            
            # Get frequencies from history output
            history = step.historyRegions['Assembly ASSEMBLY'].historyOutputs['EIGFREQ']
            frequencies = [value[1] for value in history.data]
            
            # Get node coordinates and create mapping
            instance = odb.rootAssembly.instances['PART-1-1']
            node_coords = [(node.label, node.coordinates[0]) for node in instance.nodes]
            node_coords.sort(key=lambda x: x[1])  # Sort by x-coordinate
            node_order = [node[0] for node in node_coords]  # Get ordered node labels
            
            # Get mode shapes in spatial order
            mode_shapes = []
            for i in range(1, len(step.frames)):
                frame = step.frames[i]
                displacement = frame.fieldOutputs['U'].getSubset(position=NODAL)
                
                # Create mapping of node label to U2 value
                disp_dict = {value.nodeLabel: value.data[1] for value in displacement.values}
                
                # Get U2 values in spatial order
                ordered_U2 = [disp_dict[label] for label in node_order]
                mode_shapes.append(ordered_U2)
                
            return frequencies, mode_shapes
            
        finally:
            odb.close()

    def setup_logging(self):
        logging.basicConfig(
            filename='beam_builder.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
            
    def create_materials(self):
        logging.info("Creating materials")
        model = self.mdb.models['Model-1']
        E = self.config['material']['E']
        rho = self.config['material']['rho']
        nu = self.config['material']['nu']
        
        for i in range(self.config['mesh']['n_p']):
            mat_name = f'Material-{i+1}'
            material = model.Material(name=mat_name)
            material.Density(table=((rho, ), ))
            material.Elastic(table=((E, nu), ))

    def analyze(self):
        """Execute complete beam analysis sequence"""
        self.logger.info("Starting analysis sequence")
        self.create_part()
        self.partition_beam()
        self.create_materials()
        self.create_sections()
        self.assign_sections()
        self.create_assembly()
        self.create_step()
        self.create_boundary()
        self.create_mesh()
        self.create_job()
        
        return True # self.extract_results()