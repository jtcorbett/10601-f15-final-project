function [flipped] = flipLR(images)
    r = double(:, feat(1:1024));
    g = double(:, feat(1025:2048));
    b = double(:, feat(2049:3072));
    flipped = [fliplr(r) fliplr(g) fliplr(b)]
end