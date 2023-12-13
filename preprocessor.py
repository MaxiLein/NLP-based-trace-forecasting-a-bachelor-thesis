import datetime
import json
import os
import xml.etree.ElementTree as ET

PROP_DIM = 100

# Ensure that the working directory is the file's directory
file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

# Read the XES file
year = 19 # [2012,2017,2019]
xes = ET.parse(f'../datasets/BPI_Challenge_20{year}.xes')

# Goal Format [(identifier, event_props, trace_props)] = batches of tuples of sequence-tensors 
        # each tuple is one sequence
        # Tuple = (
            #   identifier: tf.Tensor: shape=(sequence_length,), dtype=int32, 
            #   event_props: tf.Tensor: shape=(sequence_length, prop_dim), dtype=float32, 
            #   trace_props: tf.Tensor: shape=(sequence_length, prop_dim), dtype=float32
            # )
     
# Get log
log = xes.getroot()

# Extract global trace attributes
global_trace_attributes = {}
for global_trace in log.findall('.//global[@scope="trace"]'):
    for attribute in global_trace:
        if attribute.tag in [ 'date', 'int', 'float', 'boolean']:
            global_trace_attributes[attribute.get('key')] = {
                'value': attribute.get('value'),
                'type': attribute.tag
            }
        elif attribute.tag in [ 'string', 'id']:
            global_trace_attributes[attribute.get('key')] = {
                'value': attribute.get('value'),
                'type': attribute.tag,
                'map': {
                    attribute.get('value'): 0
                }
            }
        
# Sort the dictionary by keys
sorted_global_trace_attributes = {k: global_trace_attributes[k] for k in sorted(global_trace_attributes)}
# {
    # 'key' : {
        # 'value': 'default value',
        # 'type': 'type'
    # }
# }

# Extract global event attributes
global_event_attributes = {}
for global_event in log.findall('.//global[@scope="event"]'):
    for attribute in global_event:
        if attribute.tag in [ 'date', 'int', 'float', 'boolean']:
            global_event_attributes[attribute.get('key')] = {
                'value': attribute.get('value'),
                'type': attribute.tag
            }
        elif attribute.tag in [ 'string', 'id']:
            global_event_attributes[attribute.get('key')] = {
                'value': attribute.get('value'),
                'type': attribute.tag,
                'map': {
                    attribute.get('value'): 0
                }
            }
        
# Sort the dictionary by keys
sorted_global_event_attributes = {k: global_event_attributes[k] for k in sorted(global_event_attributes)}
# {
    # 'key' : {
        # 'value': 'default value',
        # 'type': 'type'
    # }
# }

# Extract the first classifier's key
classifier_key = None
classifier_default = 0
classifier = log.find('.//classifier')
if classifier is not None:
    classifier_key = classifier.get('keys')
    classifier_default = classifier.get('value')


# Map for all events
event_identifier_map = {
    'EOS': 0,
    # Default
    classifier_default : 1,
}

for event in log.findall('.//event'):
    # Get event's classifier peculiarities
    event_classifier = event.find(f".//string[@key='{classifier_key}']")
    # Check if peculiarity of classifier already in event_identifier_map
    if event_classifier is not None:
        e_val = event_classifier.get('value')
        if e_val not in event_identifier_map.keys():
            event_identifier_map[e_val] = len(event_identifier_map)

# event_identifier_map = {
#     'EOS': 0,
#     # Default
#     classifier_default : 1,
#     ... : 2,
#     ... : 3,
#     ... 
# }
        
def value_map(key, type, value, *,  event_scope=True):
    
    map = sorted_global_event_attributes if event_scope else sorted_global_trace_attributes
    
    # Maps values as described in thesis
    match type:
        case 'string' |  'id':
            # Check if value is already in map
            if map[key]['map'].get(value) is None:
                # Add value to map
                res = map[key]['map'][value] = len(map[key]['map'])
            else:
                res = map[key]['map'][value]
            return res
        case 'date':
            # Convert date to unix timestamp
            unix_timestamp = datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
            # Normalize timestamp
            res = unix_timestamp / 31_557_600_000
            return res
        case 'int' | 'float':
            return float(value)
        case 'boolean':
            return 1 if value in ['true', 'True', 'TRUE'] else -1
        case _:
            return 
            

result_identifier = []
result_trace_props = []
result_event_props = []

for trace in log.findall('trace'):
    # Extract the trace's attributes
    # Initialize the trace's attributes with defaults
    trace_props = {k : v.copy() for k,v in sorted_global_trace_attributes.items()}
    
    # For events
    events_per_trace_encoded = [] # Trace:[ Event:[Attributes], ... ]
    
    # For identifier
    event_id_per_trace = [] # Trace:[ id, ... ]
    
    # Overwrite default values
    for element in trace:
        # Trace related:
        key = element.get('key')
        if key in trace_props.keys():
            trace_props[key]['value'] = element.get('value')
            
        # Event related:
        if element.tag == 'event':
            # Extract the event's attributes
            # Initialize the event's attributes with defaults
            event_props =  {k : v.copy() for k,v in sorted_global_event_attributes.items()}
            
            
            for attribute in element:
                e_key = attribute.get('key')
                if e_key in event_props.keys():
                    event_props[e_key]['value'] = attribute.get('value')
                
                if e_key == classifier_key:
                    event_id_per_trace.append(event_identifier_map[attribute.get('value')])
            
            # Map values for Events
            event_props_encoded = [value_map(k, v['type'], v['value'], event_scope=True) for k,v in event_props.items()]
            
            # Pad event_props_encoded to PROP_DIM
            event_props_encoded.extend([0] * (PROP_DIM - len(event_props_encoded)))
            
            events_per_trace_encoded.append(event_props_encoded)
    
    # EOS Token
    eos_identfier = event_identifier_map['EOS']
    eos_trace_props = [0] * PROP_DIM
    eos_event_props = [0] * PROP_DIM
    
    # Add EOS to Identifier
    event_id_per_trace.append(eos_identfier)
    # ADD Identifier
    result_identifier += event_id_per_trace
    
    # Add EOS to Event-Props
    events_per_trace_encoded.append(eos_event_props)
    # ADD Event-Props
    result_event_props += events_per_trace_encoded
            
    
    # Map values for Traces
    trace_props_encoded = [value_map(k, v['type'], v['value'], event_scope=False) for k,v in trace_props.items()]
    
    # Pad trace_props_encoded to PROP_DIM
    trace_props_encoded.extend([0] * (PROP_DIM - len(trace_props_encoded)))
    
    # Make trace_props_encoded same shape as events_per_trace_encoded / event_id_per_trace
    trace_props_encoded = [trace_props_encoded] * len(event_id_per_trace) 
    
    # Add EOS to Trace-Props
    # trace_props_encoded.append(eos_trace_props)
    # ADD Trace-Props    
    result_trace_props += trace_props_encoded   

# Dump into json
# Maps:
with open(f'../datasets/processor_results/bpi_{year}/maps/event_map.json', 'w') as f:
    json.dump(event_identifier_map, f)
with open(f'../datasets/processor_results/bpi_{year}/maps/trace_props_map.json', 'w') as f:
    json.dump(sorted_global_trace_attributes, f)
with open(f'../datasets/processor_results/bpi_{year}/maps/event_props_map.json', 'w') as f:
    json.dump(sorted_global_event_attributes, f)

# Data:  
with open(f'../datasets/processor_results/bpi_{year}/trace/traces.json', 'w') as f:
    json.dump(result_identifier, f)
with open(f'../datasets/processor_results/bpi_{year}/trace/traces_props.json', 'w') as f:
    json.dump(result_trace_props, f)
with open(f'../datasets/processor_results/bpi_{year}/trace/events_props.json', 'w') as f:
    json.dump(result_event_props, f)

# print('sorted_global_trace_attributes \n', sorted_global_trace_attributes)
# print('sorted_global_event_attributes \n', sorted_global_event_attributes)
# print('event_identifier_map \n', event_identifier_map)


# print('result_identifier \n', len(result_identifier))
# print('result_trace_props \n', len(result_trace_props), len(result_trace_props[0]))
# print('result_event_props \n', len(result_event_props), len(result_event_props[0]))

