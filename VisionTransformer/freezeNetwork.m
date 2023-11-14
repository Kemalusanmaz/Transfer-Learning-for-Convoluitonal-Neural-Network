function net = freezeNetwork(net,args)
% netFrozen = freezeNetwork(net) sets the learning rates of all the
% parameters of the layers in the layer array |layers| to zero.
%
% netFrozen = freezeNetwork(net,LayersToIgnore=layerClassNames) specifies
% layer types to leave learning rates unchanged.

arguments
    net dlnetwork
    args.LayersToIgnore = string.empty;
end

layersToIgnore = args.LayersToIgnore;

numLayers = numel(net.Layers);

for i = 1:numLayers
    layer = net.Layers(i);

    if contains(class(layer),layersToIgnore)
        continue
    end

    p = string(properties(net.Layers(i)));
    idx = contains(p,"LearnRateFactor");

    if any(idx)
        p = p(idx);
        numProperties = numel(p);

        for j = 1:numProperties
            layer.(p(j)) = 0;
        end

        name = layer.Name;
        net = replaceLayer(net,name,layer);
    end
end

end