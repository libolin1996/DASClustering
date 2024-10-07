function data_t = generate_synthetic_data()

    nt = 1024;
    dt = 0.01;
    nf = 2 * nt;
    df = 1 / (dt * nt * 2);
    of = -nt * df;
    tt = [-nt:nt-1] * dt;
    dw = 2 * pi * df;
    nw = nt;
    ww = [0:nw-1] * dw; ww(1) = eps; ww = ww.';
    ff = of + [0:nf-1] * df; ff = ff.'; 

    % Space
    nx = 256;
    dx = 5;
    ox = -nx/2 * dx; 
    xx = ox + [0:nx-1] * dx;

    nk = nx;
    dk = 1 / (dx * nx);
    ok = -nk/2 * dk; 
    kk = ok + [0:nk-1] * dk;

    % Wavelet
    f0 = 2;
    SS = (2 .* ww.^2.5) ./ (sqrt(pi) .* (2 * pi * f0).^3) .* exp(-(ww.^2) ./ ((2 * pi * f0).^2));
    %SS = ((2 .* ww.^0.3) ./ (sqrt(pi) .* (2 * pi * f0).^3) .* exp(-(ww.^2) ./ ((2 * pi * f0).^2))).*40;
    %SS = (exp(-(ww.^2)./((2*pi*f0).^2)))./5;

    sigma = 1/(2*pi*f0);
    %SS = -1i.*ww.*exp(-sigma^2.*ww.^2/2)./300;

    % List of apparent velocities (slownesses)

    SLOWNESSES = [-0.005];
    AMPLITUDES = [1];
    ARRIVALTMS = [2];

    data_f = zeros(nw, nx);

    for ie = 1:length(SLOWNESSES)
        for ix = 1:nx
            data_f(:, ix) = data_f(:, ix) + AMPLITUDES(ie) .* SS .* exp(-1i .* ww .* (ARRIVALTMS(ie) + SLOWNESSES(ie) .* xx(ix)));
        end
    end

    % IFFT (in both time)
    data_t = 2 * real(ifftshift(ifft(data_f, 2 * nt, 1), 1));


end
