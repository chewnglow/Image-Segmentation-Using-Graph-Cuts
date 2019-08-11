%% read input image
I = imread('test.jpg');
imshow(I);

%% three lines to draw
M = imfreehand(gca,'Closed',0);
M2 = imfreehand(gca,'Closed',0);
M3 = imfreehand(gca,'Closed',0);

F = false(size(M.createMask));

%% Get three points list from three lines
P0 = M.getPosition;
P1 = M2.getPosition;
P2 = M3.getPosition;

%% merge three points list
P0 = [P0 ; P1 ; P2];

P0 = unique(round(P0),'rows');
S = sub2ind(size(I),P0(:,2),P0(:,1));
[A,B] = size(P0);
for i = 1:A
    F(int32(P0(i,2)),int32(P0(i,1))) = true;
end
figure;
imshow(F);
imwrite(F,'foreground.jpg')