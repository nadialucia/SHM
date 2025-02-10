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
import numpy as np
import os
from abaqusConstants import *
from pathlib import Path

def calc_modal(elastic_modulus, length, width, height, density, poisson, n_elements, n_eigen):
    """Calculate modal properties of beam
    Args:
        elastic_modulus: Array of elastic moduli for each segment
        length, width, height: Beam geometry
        density, poisson: Material properties
        n_elements: Number of elements for mesh
    Returns:
        frequencies: Array of natural frequencies
        mode_shapes_U2: Array of mode shapes
    """

    # Create abaqus_files directory if it doesn't exist
    abaqus_dir = Path('abaqus_files')
    abaqus_dir.mkdir(exist_ok=True)
    os.chdir(abaqus_dir)

    Mdb()

    # Create a part
    mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=2.0)
    mdb.models['Model-1'].sketches['__profile__'].Line(point1=(0.0, 0.0), point2=(length, 0.0))
    mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
        addUndoState=False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[2])
    mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='Part-1', type=DEFORMABLE_BODY)
    mdb.models['Model-1'].parts['Part-1'].BaseWire(sketch=mdb.models['Model-1'].sketches['__profile__'])
    del mdb.models['Model-1'].sketches['__profile__']

    # Get the part object
    model = mdb.models['Model-1']
    part = mdb.models['Model-1'].parts['Part-1']

    n_segments = len(elastic_modulus)

    # Initial length is 2.0m
    segment_length = length / n_segments  # We want 10 equal segments of 0.2m each

    # Create partitions
    for i in range(n_segments-1):  # We need 9 partition points for 10 segments
        remaining_length = length * (1 - i/n_segments)  # Length of current edge
        parameter = segment_length / remaining_length  # Parameter for next partition
        part.PartitionEdgeByParam(
            edges=part.edges.findAt(((0.999*length, 0.0, 0.0), )), 
            parameter=parameter)

    # Create profile
    model.RectangularProfile(a=width, b=height, name='Profile-1')

    # Create materials
    for i in range(n_segments):
        mat_name = f'Material-{i+1}'
        model.Material(name=mat_name)
        model.materials[mat_name].Density(table=((density, ), ))
        model.materials[mat_name].Elastic(table=((elastic_modulus[i], poisson), ))

    # Create sections
    for i in range(n_segments):
        model.BeamSection(
            beamSectionOffset=(0.0, 0.0), 
            consistentMassMatrix=False,
            integration=DURING_ANALYSIS,
            material=f'Material-{i+1}',
            name=f'Section-{i+1}',
            poissonRatio=0.0,
            profile='Profile-1',
            temperatureVar=LINEAR
        )

    segment_positions = [(i+.25)/n_segments * length for i in range(n_segments)]

    for i, pos in enumerate(segment_positions):
        part.SectionAssignment(
            offset=0.0,
            offsetField='',
            offsetType=MIDDLE_SURFACE,
            region=Region(
                edges=part.edges.findAt(((pos, 0.0, 0.0), ), )
            ),
            sectionName=f'Section-{i+1}',
            thicknessAssignment=FROM_SECTION
        )

    # Section orientation
    part.assignBeamSectionOrientation(method=N1_COSINES, n1=(0.0, 0.0, -1.0), region=Region(edges=part.edges))

    # Access the assembly and instance
    assembly = mdb.models['Model-1'].rootAssembly

    # Step
    assembly.DatumCsysByDefault(CARTESIAN)
    assembly.Instance(dependent=OFF, name='Part-1-1', part=mdb.models['Model-1'].parts['Part-1'])
    model.FrequencyStep(name='Step-1', numEigen=n_eigen, previous='Initial')

    
    instance = mdb.models['Model-1'].rootAssembly.instances['Part-1-1']

    # Boundary conditions
    model.PinnedBC(createStepName='Step-1', localCsys=None, name='BC-1', 
                region=Region(vertices=instance.vertices.findAt(((0.0, 0.0, 0.0), ), ((length, 0.0, 0.0), ), )))

    # Meshing
    assembly.setElementType(elemTypes=(ElemType(elemCode=B23, elemLibrary=STANDARD),), regions=Region(edges=instance.edges))
    assembly.seedPartInstance(regions=(instance,), size=0.05, deviationFactor=0.1, minSizeFactor=0.1)
    assembly.generateMesh(regions=(instance,))

    # Job
    job_name = 'Job-1'
    mdb.Job(name=job_name, model='Model-1', description='', type=ANALYSIS,
            resultsFormat=ODB, userSubroutine='', waitHours=0, waitMinutes=0)

    # Submit and wait for job
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()

    # Open ODB file from correct location
    odb_path = Path(job_name + '.odb')
    odb = session.openOdb(str(odb_path.absolute()))

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
    mode_shapes_U2 = []
    for i in range(1, len(step.frames)):
        frame = step.frames[i]
        displacement = frame.fieldOutputs['U'].getSubset(position=NODAL)
        
        # Create mapping of node label to U2 value
        disp_dict = {value.nodeLabel: value.data[1] for value in displacement.values}
        
        # Get U2 values in spatial order
        ordered_U2 = [disp_dict[label] for label in node_order]
        mode_shapes_U2.append(ordered_U2)

    frequencies = np.array(frequencies)
    mode_shapes_U2 = np.array(mode_shapes_U2)

    # Change back to original directory
    os.chdir('..')

    return frequencies, mode_shapes_U2
