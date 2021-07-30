using LibSerialPort
using Statistics
using Plots
list_ports()

portname = "/dev/cu.usbserial-A8005Afu"
baudrate = 19200

startFlg = false

xAcc = []
yAcc = []
zAcc = []


LibSerialPort.open(portname, baudrate) do sp
    while true
        if bytesavailable(sp) > 0
            msg = readline(sp)
            if occursin("Source NW Addr", msg)
                startFlg = true
                break
            elseif occursin("Data:", msg)
                startFlg = true
                break
            end
        end
    end

    while true
        if bytesavailable(sp) > 0
            msg = split(readline(sp))
            println(msg)
            try
                rawVal = parse(Int, msg[4])
                val = -(rawVal & (1 << (16 - 1))) | rawVal
                val *= 0.2
                if msg[1] == "X"
                    push!(xAcc, val)
                elseif msg[1] == "Y"
                    push!(yAcc, val)
                elseif msg[1] == "Z"
                    push!(zAcc, val)
                end
                if length(xAcc)>30
                    if var(xAcc[end-20:end]) < 0.1
                        open("evalData.txt", "w") do io
                            for idx = 1:length(xAcc)-1
                                println(io, xAcc[idx], " ", yAcc[idx], " ", zAcc[idx])
                            end
                        end
                        break
                    end
                end
            catch
                continue
            end
        end
    end
end