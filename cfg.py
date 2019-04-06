
def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] in ['convolutional', 'deconvolutional']:
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks

def print_cfg(blocks, width, height, channels):
    print('layer     filters    size              input                output');
    prev_width = width
    prev_height = height
    prev_filters = channels
    out_filters =[]
    out_widths =[]
    out_heights =[]
    ind = -1
    for block in blocks:
        ind = ind + 1
        if block['type'] in ['convolutional', 'deconvolutional']:
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = int((kernel_size-1)/2) if is_pad else 0
            if block['type'] == 'convolutional':
                width = int((prev_width + 2 * pad - kernel_size) / stride + 1)
                height = int((prev_height + 2 * pad - kernel_size) / stride + 1)
                print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            elif block['type'] == 'deconvolutional':
                width = int((prev_width - 1) * stride - 2 * pad + kernel_size)
                height = int((prev_height - 1) * stride - 2 * pad + kernel_size)
                print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'deconv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = int(prev_width/stride)
            height = int(prev_height/stride)
            print('%5d %-6s       %-2dx%2d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (ind, 'avg', prev_width, prev_height, prev_filters,  prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = int(prev_width/stride)
            height = int(prev_height/stride)
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width*stride
            height = prev_height*stride
            print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            layerlen = len(layers)
            assert(layerlen >= 1)
            prev_width = out_widths[layers[0]]
            prev_height = out_heights[layers[0]]
            prev_filters = out_filters[layers[0]]
            prtstr = ('%5d %-6s  %d' % (ind, 'route', layers[0]))
            if layerlen > 1:
                for i in range(1, layerlen):
                    prtstr += (' %d' % layers[i])
                    assert (prev_width == out_widths[layers[i]])
                    assert (prev_height == out_heights[layers[i]])
                    prev_filters += out_filters[layers[i]]
            print(prtstr)
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'outlayer':
            print('%5d %-6s' % (ind, block['type']))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id+ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters,  filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))

if __name__ == '__main__':
    import sys
    blocks = parse_cfg('cfg/yolov3.cfg')
    if len(sys.argv) == 2:
        blocks = parse_cfg(sys.argv[1])
    print_cfg(blocks)
