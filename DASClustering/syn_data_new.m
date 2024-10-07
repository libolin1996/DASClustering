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
    SS1 = ((2 .* ww.^0.8) ./ (sqrt(pi) .* (2 * pi * f0).^3) .* exp(-(ww.^2) ./ ((2 * pi * f0).^2))).*10;
    SS2 = (exp(-(ww.^2)./((2*pi*f0).^2)))./30;

    sigma = 1/(2*pi*f0);
    SS1 = -1i.*ww.*exp(-sigma^2.*ww.^2/2)./1000;
    %SS2 = -1i.*ww.*exp(-sigma^2.*ww.^2/2)./1000;

    S1 = linspace(0.002,0.0055,nx);
    S2 = S1; S3 = S1; S4 = S1;
    SLOWNESSES1 = [S1; S2; S3; S4]; 
    AMPLITUDES1 = [1; -0.5; 0.95; -0.5]; 
    x1=3;
    AMPLITUDES1 = [x1; -x1; x1; -x1];
    ARRIVALTMS1 = [-7.5; -2.5; 2.5; 7];  
    S5 = linspace(-0.004,-0.009,nx);
    S6 = S5; 
    SLOWNESSES2 = [S5; S6];
    AMPLITUDES2 = [-1; -0.95];
    AMPLITUDES2 = [0.16; 0.16];
    ARRIVALTMS2 = [-2; 6.5];

    SLOWNESSES = [SLOWNESSES1; SLOWNESSES2];
    AMPLITUDES = [AMPLITUDES1; AMPLITUDES2];
    ARRIVALTMS = [ARRIVALTMS1; ARRIVALTMS2];

    data_f = zeros(nw, nx);

    for ie = 1:size(SLOWNESSES1,1)
        for ix = 1:nx
            data_f(:, ix) = data_f(:, ix) + AMPLITUDES1(ie) .* SS1 .* exp(-1i .* ww .* (ARRIVALTMS1(ie) + SLOWNESSES1(ie,ix) .* xx(ix)));
        end
    end

    for ie = 1:size(SLOWNESSES2,1)
        for ix = 1:nx
            data_f(:, ix) = data_f(:, ix) + AMPLITUDES2(ie) .* SS2 .* exp(-1i .* ww .* (ARRIVALTMS2(ie) + SLOWNESSES2(ie,ix) .* xx(ix)));
        end
    end

    % IFFT (in both time)
    data_t = 2 * real(ifftshift(ifft(data_f, 2 * nt, 1), 1));

end
