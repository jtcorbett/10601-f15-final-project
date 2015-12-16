function [flipped] = flipLR(images)
    r = images(:, 1:1024);
    g = images(:, 1025:2048);
    b = images(:, 2049:3072);
    flipped = [fliplr(r) fliplr(g) fliplr(b)];
end