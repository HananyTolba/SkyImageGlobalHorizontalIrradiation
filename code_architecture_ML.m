%% Charger préalablement DateTimeCom et ImGHIrec2
clear all; close all; clc;
load ImGHIrec2
DTC =DateTimeCom(1:13108);

InPx  = [0:5]; % InP(1) doit être égal à 0 puis attibuer un délai en minute par rapport au 0 pour la/les entrée(s) et la/les sortie(s)
OutPx = [10 20]; % InP et OutP ne doivent pas forcément avoir la même longueur
IOPx = [InPx OutPx];
for i =1:length(IOPx)
    Z{i} = datefind(DTC-minutes(IOPx(i)), DTC, 0);
end
RefP = intersectm(Z{:});
%InP = RefP + InPx;

%% Data brutes
DataU = reshape(ImGHIrec2,[],size(ImGHIrec2,3));

%% Input data
for i=1:length(RefP)
    for ii=1:length(InPx)
        DataI(:,ii,i) = DataU(:,RefP(i)+InPx(ii));
    end
end
DataI=reshape(DataI,[],length(RefP));

%% Output data
for i=1:length(RefP)
    for ii=1:length(OutPx)
        DataO(:,ii,i) = DataU(:,RefP(i)+OutPx(ii));
    end
end
DataO=reshape(DataO,[],length(RefP));


function [A,ia,varargout] = intersectm(A,varargin)
varargout = cell(size(varargin));
ia = 1:numel(A);
for ii = 1:numel(varargin)
    [A,ixa,varargout{ii}] = intersect(A,varargin{ii});
    ia = ia(ixa);
    for jj = 1:ii-1
        varargout{jj} = varargout{jj}(ixa);
    end
end
end
